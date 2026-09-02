// Copyright © 2024 Viper Contributors.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file.

package viper

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/fsnotify/fsnotify"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// reloadTimeout is a safety bound for the deterministic, channel-driven
// waits below; timing itself is never asserted.
const reloadTimeout = 10 * time.Second

type reloadFailure struct {
	err  error
	file string
}

func waitForSignal[T any](t *testing.T, ch <-chan T, what string) T {
	t.Helper()

	select {
	case x := <-ch:
		return x
	case <-time.After(reloadTimeout):
		t.Fatalf("timed out waiting for %s", what)
		var zero T
		return zero
	}
}

func assertNoFailure(t *testing.T, ch <-chan reloadFailure) {
	t.Helper()

	select {
	case f := <-ch:
		t.Fatalf("unexpected reload failure: %v", f.err)
	default:
	}
}

// newTransactionalViper creates a temp config file and a Viper instance with
// transactional reload enabled and the initial configuration loaded as the
// last good baseline.
func newTransactionalViper(t *testing.T, initial string) (*Viper, string) {
	t.Helper()

	watchDir := t.TempDir()
	configFile := filepath.Join(watchDir, "config.yaml")
	require.NoError(t, os.WriteFile(configFile, []byte(initial), 0o640))

	v := NewWithOptions(TransactionalReload())
	v.SetConfigFile(configFile)
	require.NoError(t, v.ReadInConfig())

	return v, configFile
}

// writeConfigAtomic writes content via a temp file + rename, mimicking the
// atomic save behavior of editors.
func writeConfigAtomic(t *testing.T, configFile, content string) {
	t.Helper()

	tmp, err := os.CreateTemp(filepath.Dir(configFile), ".config-*")
	require.NoError(t, err)
	tmpName := tmp.Name()
	require.NoError(t, tmp.Close())
	require.NoError(t, os.WriteFile(tmpName, []byte(content), 0o640))
	require.NoError(t, os.Rename(tmpName, configFile))
}

func TestTransactionalReloadSuccess(t *testing.T) {
	v, configFile := newTransactionalViper(t, "foo: bar\n")
	require.Equal(t, "bar", v.GetString("foo"))

	changed := make(chan fsnotify.Event, 8)
	failed := make(chan reloadFailure, 8)
	v.OnConfigChange(func(e fsnotify.Event) { changed <- e })
	v.OnConfigReloadError(func(err error, file string) { failed <- reloadFailure{err: err, file: file} })

	v.WatchConfig()

	require.NoError(t, os.WriteFile(configFile, []byte("foo: baz\n"), 0o640))

	waitForSignal(t, changed, "config change callback")

	require.Equal(t, "baz", v.GetString("foo"))
	assertNoFailure(t, failed)
}

func TestTransactionalReloadAtomicReplace(t *testing.T) {
	v, configFile := newTransactionalViper(t, "foo: bar\n")

	changed := make(chan fsnotify.Event, 8)
	failed := make(chan reloadFailure, 8)
	v.OnConfigChange(func(e fsnotify.Event) { changed <- e })
	v.OnConfigReloadError(func(err error, file string) { failed <- reloadFailure{err: err, file: file} })

	v.WatchConfig()

	// Editor-style atomic save: temp file + rename over the watched file.
	writeConfigAtomic(t, configFile, "foo: renamed\n")

	waitForSignal(t, changed, "config change callback after atomic replace")

	require.Equal(t, "renamed", v.GetString("foo"))
	assertNoFailure(t, failed)
}

func TestTransactionalReloadParseErrorKeepsLastGood(t *testing.T) {
	v, configFile := newTransactionalViper(t, "foo: bar\nport: 8080\n")

	changed := make(chan fsnotify.Event, 8)
	failed := make(chan reloadFailure, 8)
	v.OnConfigChange(func(e fsnotify.Event) { changed <- e })
	v.OnConfigReloadError(func(err error, file string) { failed <- reloadFailure{err: err, file: file} })

	v.WatchConfig()

	// Write syntactically invalid YAML: the editor has not finished saving.
	require.NoError(t, os.WriteFile(configFile, []byte("foo: [unterminated\n"), 0o640))

	failure := waitForSignal(t, failed, "reload failure after parse error")

	var parseErr ConfigParseError
	require.True(t, errors.As(failure.err, &parseErr), "expected ConfigParseError, got %T: %v", failure.err, failure.err)
	assert.Equal(t, configFile, failure.file, "failure must report the changed file")

	// Last good configuration stays active for every read path.
	require.Equal(t, "bar", v.GetString("foo"))
	require.Equal(t, 8080, v.GetInt("port"))
	require.Equal(t, "bar", v.AllSettings()["foo"])

	select {
	case <-changed:
		t.Fatal("change callback must not fire on a failed reload")
	default:
	}

	// The next valid change publishes normally.
	require.NoError(t, os.WriteFile(configFile, []byte("foo: baz\nport: 9090\n"), 0o640))

	waitForSignal(t, changed, "config change callback after recovery")

	require.Equal(t, "baz", v.GetString("foo"))
	require.Equal(t, 9090, v.GetInt("port"))
	assertNoFailure(t, failed)
}

func TestTransactionalReloadValidationErrorKeepsLastGood(t *testing.T) {
	v, configFile := newTransactionalViper(t, "foo: bar\nport: 8080\n")

	type configShape struct {
		Foo  string `mapstructure:"foo"`
		Port int    `mapstructure:"port"`
	}

	observed := make(chan [2]int, 8) // [candidate port, live port]
	v.AddConfigValidator(func(c *Viper) error {
		// The validator reads the fully merged candidate configuration;
		// Unmarshal on the candidate exercises the whole merged view.
		var probe configShape
		if err := c.Unmarshal(&probe); err != nil {
			return err
		}

		observed <- [2]int{c.GetInt("port"), v.GetInt("port")}

		if c.GetInt("port") <= 0 {
			return fmt.Errorf("port must be positive, got %d", c.GetInt("port"))
		}

		return nil
	})

	changed := make(chan fsnotify.Event, 8)
	failed := make(chan reloadFailure, 8)
	v.OnConfigChange(func(e fsnotify.Event) { changed <- e })
	v.OnConfigReloadError(func(err error, file string) { failed <- reloadFailure{err: err, file: file} })

	v.WatchConfig()

	// Parseable but violating the business constraint.
	require.NoError(t, os.WriteFile(configFile, []byte("foo: baz\nport: -1\n"), 0o640))

	// Validator sees the candidate value while the live instance keeps
	// serving the last good value.
	ports := waitForSignal(t, observed, "validator invocation for invalid config")
	require.Equal(t, [2]int{-1, 8080}, ports, "validator reads candidate; live instance stays last good")

	failure := waitForSignal(t, failed, "reload failure after validation error")

	var validationErr ConfigValidationError
	require.True(t, errors.As(failure.err, &validationErr), "expected ConfigValidationError, got %T: %v", failure.err, failure.err)
	assert.Equal(t, configFile, failure.file)

	// Last good configuration stays active.
	require.Equal(t, "bar", v.GetString("foo"))
	require.Equal(t, 8080, v.GetInt("port"))

	var live configShape
	require.NoError(t, v.Unmarshal(&live))
	require.Equal(t, configShape{Foo: "bar", Port: 8080}, live)

	select {
	case <-changed:
		t.Fatal("change callback must not fire on a rejected candidate")
	default:
	}

	// The next valid change publishes normally.
	require.NoError(t, os.WriteFile(configFile, []byte("foo: baz\nport: 9090\n"), 0o640))

	waitForSignal(t, changed, "config change callback after recovery")

	require.Equal(t, "baz", v.GetString("foo"))
	require.Equal(t, 9090, v.GetInt("port"))
	assertNoFailure(t, failed)
}

func TestTransactionalReloadStaleResultDoesNotOverwrite(t *testing.T) {
	// Drive the coordinator directly (without fsnotify) so reload generation
	// ordering is fully deterministic.
	v, configFile := newTransactionalViper(t, "foo: one\n")

	gate := make(chan struct{})
	var entered atomic.Int32
	enteredValues := make(chan string, 8)
	v.AddConfigValidator(func(c *Viper) error {
		n := entered.Add(1)
		enteredValues <- c.GetString("foo")
		if n == 1 {
			// The first (older) validation blocks until the test releases it.
			<-gate
		}
		return nil
	})

	var changeCount atomic.Int32
	changed := make(chan fsnotify.Event, 8)
	failed := make(chan reloadFailure, 8)
	v.OnConfigChange(func(e fsnotify.Event) {
		changeCount.Add(1)
		changed <- e
	})
	v.OnConfigReloadError(func(err error, file string) { failed <- reloadFailure{err: err, file: file} })

	reloader := newTxReloader(v, configFile)
	reloader.start()
	defer reloader.stop()

	// First change: reload starts, reads "two" and blocks inside validation.
	require.NoError(t, os.WriteFile(configFile, []byte("foo: two\n"), 0o640))
	reloader.notify(fsnotify.Event{Name: configFile, Op: fsnotify.Write})
	require.Equal(t, "two", waitForSignal(t, enteredValues, "first validator invocation"))

	// Second change arrives while the older validation is still running.
	require.NoError(t, os.WriteFile(configFile, []byte("foo: three\n"), 0o640))
	reloader.notify(fsnotify.Event{Name: configFile, Op: fsnotify.Write})

	// Release the stale validation: it must be discarded without publishing
	// or notifying; the newer reload then reads "three" and publishes.
	close(gate)

	waitForSignal(t, changed, "change callback for the newer reload")
	require.Equal(t, "three", waitForSignal(t, enteredValues, "second validator invocation"))

	require.Equal(t, "three", v.GetString("foo"))
	require.Equal(t, int32(1), changeCount.Load(), "the stale reload must not publish or notify")
	assertNoFailure(t, failed)
}

func TestTransactionalReloadCoalescesBurst(t *testing.T) {
	v, configFile := newTransactionalViper(t, "foo: v0\n")

	const writes = 25

	settled := make(chan struct{})
	var settleOnce sync.Once
	failed := make(chan reloadFailure, 8)
	v.OnConfigChange(func(fsnotify.Event) {
		if v.GetString("foo") == fmt.Sprintf("v%d", writes) {
			settleOnce.Do(func() { close(settled) })
		}
	})
	v.OnConfigReloadError(func(err error, file string) { failed <- reloadFailure{err: err, file: file} })

	v.WatchConfig()

	// Rapid consecutive writes (as editors and atomic saves produce); events
	// are coalesced but the latest state must never be lost.
	for i := 1; i <= writes; i++ {
		require.NoError(t, os.WriteFile(configFile, []byte(fmt.Sprintf("foo: v%d\n", i)), 0o640))
	}

	waitForSignal(t, settled, "final state after burst of writes")

	require.Equal(t, fmt.Sprintf("v%d", writes), v.GetString("foo"))
	assertNoFailure(t, failed)
}

func TestTransactionalReloadConcurrentReads(t *testing.T) {
	v, configFile := newTransactionalViper(t, "foo: bar\n")

	validValues := map[string]bool{"bar": true, "baz": true, "qux": true}

	failed := make(chan reloadFailure, 8)
	v.OnConfigReloadError(func(err error, file string) { failed <- reloadFailure{err: err, file: file} })

	type configShape struct {
		Foo string `mapstructure:"foo"`
	}

	stopReaders := make(chan struct{})
	var readers sync.WaitGroup
	problems := make(chan string, 1024)

	for i := 0; i < 16; i++ {
		readers.Add(1)
		go func() {
			defer readers.Done()

			for {
				select {
				case <-stopReaders:
					return
				default:
				}

				foo := v.GetString("foo")
				if !validValues[foo] {
					problems <- fmt.Sprintf("Get observed invalid value %q", foo)
					return
				}

				settings := v.AllSettings()
				if settingsFoo, _ := settings["foo"].(string); !validValues[settingsFoo] {
					problems <- fmt.Sprintf("AllSettings observed invalid value %q", settingsFoo)
					return
				}

				var decoded configShape
				if err := v.Unmarshal(&decoded); err != nil {
					problems <- fmt.Sprintf("Unmarshal failed: %v", err)
					return
				}
				if !validValues[decoded.Foo] {
					problems <- fmt.Sprintf("Unmarshal observed invalid value %q", decoded.Foo)
					return
				}
			}
		}()
	}

	changed := make(chan fsnotify.Event, 8)
	v.OnConfigChange(func(e fsnotify.Event) { changed <- e })

	v.WatchConfig()

	// Valid change: published.
	require.NoError(t, os.WriteFile(configFile, []byte("foo: baz\n"), 0o640))
	waitForSignal(t, changed, "change for valid update")

	// Invalid change (still being written / syntactically broken): rejected,
	// readers must keep seeing valid, last good values.
	require.NoError(t, os.WriteFile(configFile, []byte("foo: [unterminated\n"), 0o640))
	waitForSignal(t, failed, "failure for broken update")

	// Another valid change after the failure: published.
	require.NoError(t, os.WriteFile(configFile, []byte("foo: qux\n"), 0o640))
	waitForSignal(t, changed, "change for valid update after failure")

	close(stopReaders)
	readers.Wait()

	select {
	case problem := <-problems:
		t.Fatalf("concurrent reader observed invalid state: %s", problem)
	default:
	}

	require.Equal(t, "qux", v.GetString("foo"))
}

func TestTransactionalReloadDisabledByDefault(t *testing.T) {
	// Without the option, WatchConfig keeps its historical behavior:
	// validators are ignored and every change event fires the change handler.
	watchDir := t.TempDir()
	configFile := filepath.Join(watchDir, "config.yaml")
	require.NoError(t, os.WriteFile(configFile, []byte("foo: bar\n"), 0o640))

	v := New()
	v.SetConfigFile(configFile)
	require.NoError(t, v.ReadInConfig())
	require.False(t, v.transactionalReload)

	// Registered but must be ignored in legacy mode.
	v.AddConfigValidator(func(c *Viper) error {
		return errors.New("validators are not used without transactional reload")
	})

	changed := make(chan fsnotify.Event, 8)
	v.OnConfigChange(func(e fsnotify.Event) { changed <- e })

	v.WatchConfig()

	require.NoError(t, os.WriteFile(configFile, []byte("foo: baz\n"), 0o640))

	waitForSignal(t, changed, "legacy change callback")

	require.Equal(t, "baz", v.GetString("foo"), "legacy reload behavior must be unchanged")
}
