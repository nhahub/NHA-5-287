# Changelog

## v0 (NexusML)
- Renamed from "ds_agent" / "Meridian": package `nexusml`, environment variables `NEXUSML_*` (was `DS_AGENT_*`).
- Fixed: the global font rule overrode the icon font, which can turn sidebar icons into plain words. Icons are now excluded.
- Removed unused imports, the unreferenced `overview.py` and `chatstore.py`, and the per-version notes (history is below).
- Matplotlib boxplot uses `orientation=` (needs matplotlib 3.10+).
- Plainer wording in the interface and reports ("CV score" instead of "honest score").
- One README, one architecture note.
- Fixed (images): an uploaded image zip stored only its CSV, with relative image paths pointing at nothing, so every image became missing data and models scored at chance (F1 0.17 on a 3-class set) without any error. The stored table now holds absolute paths, datasets saved the old way are located again automatically, and a run stops with a clear message if most image files cannot be found.
- Images: common loader for all backends (EXIF rotation, transparency on white, 16-bit and palette files), a 4x4 colour layout feature, and a new rotation-invariant outline-shape backend (segmentation, compactness, solidity, Hu moments, outline harmonics) offered as separate candidates. On a shapes-only test set held-out F1 went from about 0.5 to about 0.97.
- Added: image preprocessing recipes (resize with squash / fit / fill, blur, sharpen, median denoise, histogram equalise, contrast stretch, grayscale, Sobel edge detection). Chosen on New analysis > Image options with a before/after preview, stored with the model, and applied identically to new photos. CLI/API: `image_prep=["blur:1.5", "edges"]`.
- Added: deep-network features. ResNet-18 and ResNet-50 (ImageNet weights) feed LogisticRegression / SVC / Ridge candidates next to the hand-crafted ones; batched, cached, GPU used when present. Choose Automatic / Off / ResNet-18 / ResNet-50 / Both per run. Offline: set `NEXUSML_DEEP_WEIGHTS` to a saved `.pth` file. Needs torch + torchvision (`Dockerfile.multimodal`); without them the hand-crafted models run as before.
- Added: Google Gemini as the LLM provider (OpenAI-compatible endpoint). `GEMINI_API_KEY` selects it and the low-cost `gemini-3.1-flash-lite` model; Groq still works with `GROQ_API_KEY`.
- Added: Data page > Combine tables. Pick two or more datasets, review the proposed joins, approve, and get one combined dataset. Before this, multi-table joins only worked from the CLI.

## Earlier work (as ds_agent)
- v7: multi-page app, assistant with 15 validated tools, workspace (datasets, saved chats), background jobs, per-session user identity.
- v6: parallel agent crew, goal-driven AutoML.
- v5: leaderboard, 13-stage graph, NLP and vision pipelines.
- v4: documents/images as input, MLflow integration, chat history.
