# Instructions for future agents

1. Read `docs/HANDOFF.md`, `docs/METHODS.md` and `docs/VALIDATION.md` first.
2. This is a behavior-preserving scientific-code refactor. Keep historical
   result provenance and distinguish old/new sampling kernels and score scales.
3. Do not download/run ESM2 or ProstT5 locally, or train an encoder, unless the
   user explicitly asks. GPU execution belongs on RunPod. CPU tests use fakes.
4. Do not use ground truth during inference, proposal selection, chain selection
   or hyperparameter tuning. Evaluation is a separate post-inference stage.
5. Infer AA/3Di node correspondence by topology, not by matching node numbers.
   Glu/Gln deliberately pairs AA Node168 with 3Di Node16; EF-Tu has its own mapping.
6. Keep `examples/` immutable. Write outputs/caches to ignored `results/` and
   `.cache/`. Never modify the Desktop originals as part of this checkout.
7. Preserve old `runpod_*` import paths/config factories when moving code.
   Test checkpoint resume and validate inputs without loading GPU models.
8. Run `python -m unittest discover -s tests -v` and the CPU example replay
   after changes. Update docs if numerical behavior, defaults or data changes.
9. New dependencies/model downloads require RunPod instructions, not local
   installation attempts. Do not silently bypass model-loading security checks.
10. This handoff contains no active task queue. Follow the user's new request;
    do not resume previously discussed jobs or extensions automatically.
    Historical workflow examples and limitations are context, not assignments.
