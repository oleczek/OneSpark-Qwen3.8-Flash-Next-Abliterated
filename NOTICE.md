# Attribution and licenses

| Source | Revision | Contribution and license |
| --- | --- | --- |
| [azampatti/GB10-3.8-Flash-Next](https://github.com/azampatti/GB10-3.8-Flash-Next) | `2dee9f3ba6ef00d5b46411ba5cb1fae047d77015` | Launch recipe, FP8/PLE-off patchers and benchmark prompts; Apache-2.0 |
| [deathbyorderfill/flashnext-one-spark](https://github.com/deathbyorderfill/flashnext-one-spark) | `55c820de3ea665f25037388727472bedb0a9ead8` | HashK builder and patch generators; MIT |
| [SGLang](https://github.com/sgl-project/sglang) | Runtime sources supplied through deathbyorderfill | Model and QSA runtime files in `patches/`; Apache-2.0 |
| [FlashAttention](https://github.com/Dao-AILab/flash-attention) | Patched source supplied through deathbyorderfill | `patches/flash_fwd.py`; BSD-3-Clause |

The `patches/` tree is unchanged from the cited deathbyorderfill revision.
Local changes add pinned downloads, file verification, safer launch/stop,
authenticated API tests and patch validation. The HashK builder adds in-place
means, corrected head-boundary assignment and verified atomic artifact output.
Runtime model changes are generated under `.state/`.

[LICENSE](LICENSE) covers the port's contributions and Apache-derived portions.
Third-party terms remain in effect. Full MIT/BSD notices and FlashAttention
authors are in [LICENSES/](LICENSES/); retain them and the source copyright headers.

The model and abliteration come from
[dealignai](https://huggingface.co/dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4)
and [Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), with their own terms.
Model weights and HashK artifacts are not bundled. This port is not affiliated
with the upstream authors or NVIDIA.
