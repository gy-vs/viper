// Copyright © 2024 Viper Contributors.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file.

package viper

import (
	"fmt"
	"sync"

	"github.com/fsnotify/fsnotify"
)

// ConfigValidator checks a candidate configuration during a transactional
// reload.
//
// The candidate is an isolated, fully merged read-only view: reads from it
// ([Viper.Get], [Viper.Unmarshal], [Viper.AllSettings], ...) reflect the new
// configuration, while concurrent readers of the live Viper instance still
// observe the last good configuration. No external locking is required.
//
// Returning a non-nil error aborts the reload: the candidate is discarded, the
// last good configuration stays active, the change handler is not invoked and
// the error (wrapped in [ConfigValidationError]) is reported to the handler
// registered with [Viper.OnConfigReloadError].
//
// Validators must not mutate the candidate and should return in a bounded
// amount of time: while a validator runs, subsequent file changes are
// coalesced and processed once it returns.
type ConfigValidator func(candidate *Viper) error

// ConfigValidationError wraps the error returned by a [ConfigValidator] that
// rejected a candidate configuration during a transactional reload.
type ConfigValidationError struct {
	err error
}

// Error returns the formatted validation error.
func (e ConfigValidationError) Error() string {
	return fmt.Sprintf("config validation failed: %s", e.err.Error())
}

// Unwrap returns the error returned by the validator.
func (e ConfigValidationError) Unwrap() error {
	return e.err
}

// TransactionalReload enables transactional, atomic configuration reloads.
//
// When enabled, [Viper.WatchConfig] applies file changes in isolated
// transactions: a change is read, merged and validated on a copy of the
// configuration, and published atomically only when every step succeeds.
//
// On failure (the file cannot be read/parsed or a registered
// [ConfigValidator] rejects the candidate) the last good configuration stays
// active, the change handler does not run, and the failure is reported through
// [Viper.OnConfigReloadError]. The next valid change is applied normally.
//
// Bursts of file events (for example the multiple writes an editor produces,
// or rename/replace atomic saves) are coalesced into a single reload of the
// latest file state, and a reload that finishes after a newer change was
// observed is discarded without affecting the active configuration.
//
// The mode is opt-in: when disabled (the default), [Viper.WatchConfig] keeps
// its historical behavior. Configuration source precedence, aliases,
// environment variables and remote configuration semantics are unchanged.
func TransactionalReload() Option {
	return optionFunc(func(v *Viper) {
		v.transactionalReload = true
	})
}

// AddConfigValidator registers a validation function run against candidate
// configurations during transactional reloads.
//
// Validators run in registration order, after the changed configuration file
// has been read and merged, before the result is published. If any validator
// returns an error, the reload is aborted, the previous (last good)
// configuration remains active and the error is reported to the handler
// registered with [Viper.OnConfigReloadError].
//
// Validators only take effect when transactional reload is enabled (see
// [TransactionalReload]).
func AddConfigValidator(validator ConfigValidator) { v.AddConfigValidator(validator) }

// AddConfigValidator registers a validation function run against candidate
// configurations during transactional reloads.
func (v *Viper) AddConfigValidator(validator ConfigValidator) {
	if validator == nil {
		return
	}

	v.hooksMu.Lock()
	defer v.hooksMu.Unlock()

	v.configValidators = append(v.configValidators, validator)
}

// OnConfigReloadError sets the handler called when a transactional reload
// fails, either because the configuration could not be read/parsed or because
// a registered [ConfigValidator] rejected it. The handler receives the error
// and the path of the configuration file that triggered the reload.
//
// The active configuration is unchanged when the handler runs: reads still
// return the last good configuration. The handler is invoked outside of any
// internal lock, so it may safely read the configuration.
//
// Failures are not retried automatically: the next file change triggers a new
// transaction.
//
// The handler is only used when transactional reload is enabled (see
// [TransactionalReload]).
func OnConfigReloadError(run func(err error, file string)) { v.OnConfigReloadError(run) }

// OnConfigReloadError sets the handler called when a transactional reload
// fails.
func (v *Viper) OnConfigReloadError(run func(err error, file string)) {
	v.hooksMu.Lock()
	defer v.hooksMu.Unlock()

	v.onReloadError = run
}

// txReloader coordinates transactional config reloads for a single
// [Viper.WatchConfig] invocation.
//
// File events from the watcher goroutine are handed to [txReloader.notify],
// which coalesces bursts into a single pending reload and wakes a single
// coordinator goroutine. Each reload is staged on an isolated clone and only
// published if it is still current when it completes: a reload whose epoch is
// older than the latest observed change is dropped entirely (no publish, no
// notification), so a slow validation can never overwrite a newer result.
type txReloader struct {
	v          *Viper
	configFile string

	trigger chan struct{}
	stopCh  chan struct{}
	done    chan struct{}

	mu    sync.Mutex
	epoch uint64
	dirty bool
	event fsnotify.Event
}

func newTxReloader(v *Viper, configFile string) *txReloader {
	return &txReloader{
		v:          v,
		configFile: configFile,
		trigger:    make(chan struct{}, 1),
		stopCh:     make(chan struct{}),
		done:       make(chan struct{}),
	}
}

// start launches the reload coordinator goroutine.
func (r *txReloader) start() {
	go r.loop()
}

// stop signals the coordinator to shut down and waits for it to finish. It is
// called at most once, from the watcher event loop goroutine.
func (r *txReloader) stop() {
	select {
	case <-r.stopCh:
		// already closed
	default:
		close(r.stopCh)
	}
	<-r.done
}

// notify records a file change and wakes the coordinator. It never blocks:
// notifications that arrive while a reload is already queued or in flight are
// coalesced into the next reload of the latest file state.
func (r *txReloader) notify(event fsnotify.Event) {
	r.mu.Lock()
	r.epoch++
	r.dirty = true
	r.event = event
	r.mu.Unlock()

	select {
	case r.trigger <- struct{}{}:
	default:
	}
}

// loop is the coordinator goroutine. It drains coalesced events and applies
// them one by one until the watcher shuts down.
func (r *txReloader) loop() {
	defer close(r.done)

	for {
		select {
		case <-r.stopCh:
			return
		case <-r.trigger:
		}

		for {
			r.mu.Lock()
			if !r.dirty {
				r.mu.Unlock()
				break
			}
			event := r.event
			gen := r.epoch
			r.dirty = false
			r.mu.Unlock()

			r.apply(event, gen)
		}
	}
}

// apply stages, validates and (if still current) publishes a single reload.
func (r *txReloader) apply(event fsnotify.Event, gen uint64) {
	v := r.v

	// Stage the reload on an isolated clone: reading and parsing the file and
	// running validators never touches the live configuration or races with
	// concurrent readers.
	candidate := v.cloneForReload()

	err := candidate.ReadInConfig()
	if err == nil {
		v.hooksMu.RLock()
		validators := v.configValidators
		v.hooksMu.RUnlock()

		for _, validator := range validators {
			if vErr := validator(candidate); vErr != nil {
				err = ConfigValidationError{err: vErr}
				break
			}
		}
	}

	// Decide and publish atomically with respect to newer notifications: if a
	// newer change arrived while this reload was in flight, its result is stale
	// and dropped entirely (no publish, no notification); the next iteration
	// reloads the latest file state instead.
	var (
		publish   bool
		reloadErr error
	)

	r.mu.Lock()
	if gen == r.epoch {
		if err != nil {
			reloadErr = err
		} else {
			v.configMu.Lock()
			v.config = candidate.config
			v.configMu.Unlock()
			publish = true
		}
	}
	r.mu.Unlock()

	if reloadErr != nil {
		v.logger.Error(fmt.Sprintf("transactional config reload failed: %s", reloadErr))

		v.hooksMu.RLock()
		handler := v.onReloadError
		v.hooksMu.RUnlock()

		if handler != nil {
			handler(reloadErr, r.configFile)
		}
		return
	}

	if publish {
		v.fireConfigChange(event)
	}
}

// cloneForReload returns an isolated Viper instance used to stage a
// transactional reload. Only immutable infrastructure (filesystem, finder,
// codec registries, logger) is shared; every configuration layer is copied, so
// reading, merging and validating the candidate cannot affect the live
// instance or race with concurrent readers.
//
// The candidate starts with an empty config layer: [Viper.ReadInConfig]
// replaces it entirely with the parsed file, preserving the source precedence
// rules of the live instance.
func (v *Viper) cloneForReload() *Viper {
	c := &Viper{
		keyDelim:               v.keyDelim,
		fs:                     v.fs,
		finder:                 v.finder,
		configName:             v.configName,
		configFile:             v.configFile,
		configType:             v.configType,
		configPermissions:      v.configPermissions,
		envPrefix:              v.envPrefix,
		automaticEnvApplied:    v.automaticEnvApplied,
		envKeyReplacer:         v.envKeyReplacer,
		allowEmptyEnv:          v.allowEmptyEnv,
		typeByDefValue:         v.typeByDefValue,
		logger:                 v.logger,
		encoderRegistry:        v.encoderRegistry,
		decoderRegistry:        v.decoderRegistry,
		decodeHook:             v.decodeHook,
		experimentalFinder:     v.experimentalFinder,
		experimentalBindStruct: v.experimentalBindStruct,

		config:   make(map[string]any),
		override: copyAndInsensitiviseMap(v.override),
		defaults: copyAndInsensitiviseMap(v.defaults),
		kvstore:  copyAndInsensitiviseMap(v.kvstore),
		aliases:  cloneStringMap(v.aliases),
		pflags:   cloneFlagMap(v.pflags),
		env:      cloneEnvMap(v.env),

		parents:         append([]string(nil), v.parents...),
		configPaths:     append([]string(nil), v.configPaths...),
		remoteProviders: append([]*defaultRemoteProvider(nil), v.remoteProviders...),
	}

	return c
}

func cloneStringMap(src map[string]string) map[string]string {
	dst := make(map[string]string, len(src))
	for k, v := range src {
		dst[k] = v
	}
	return dst
}

func cloneFlagMap(src map[string]FlagValue) map[string]FlagValue {
	dst := make(map[string]FlagValue, len(src))
	for k, v := range src {
		dst[k] = v
	}
	return dst
}

func cloneEnvMap(src map[string][]string) map[string][]string {
	dst := make(map[string][]string, len(src))
	for k, v := range src {
		dst[k] = append([]string(nil), v...)
	}
	return dst
}
