# Future question: learned data-consistency weights

Recorded, not investigated. This is separate from the thin-structure study and must not be mixed
into it.

**Observation.** In all ten mixed-family networks (five control, five intervention) the learned
data-consistency weights lie between 0.008 and 0.027 in every round and are nearly the same for 16
and 64 sensors. They were initialised at 0.01. Tikhonov's selected weight on the same training
split is 0.01 at 14 dB SNR for both sensor counts, but 0.001 at 40 dB SNR and 1e-7 without noise
with 64 sensors (`report/mixed_phantom_results.json`). The networks have one weight per round for
all noise levels.

**Question.** Does insufficient adaptation of the learned data-consistency weights explain the large
gap to Tikhonov at low noise and high sensor count?

**Where to look.** Primarily at finite low noise, 26 and 40 dB SNR, where the networks are level
with or behind Tikhonov with 64 sensors on some image families. The noiseless case is an inverse
crime (Tikhonov inverts the operator that generated the data) and is weak evidence on its own.

**Status.** The weights were read from the checkpoints during an audit. No experiment has been
designed or run.
