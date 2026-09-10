# CJK fonts for egui

egui needs individual **TTF** files. Google’s Noto Sans CJK ships as a **TTC** collection, so we extract faces locally (not committed — each face is ~16 MB).

| File | Script / region |
|------|-----------------|
| `NotoSansCJKsc-Regular.ttf` | Simplified Chinese (SC) |
| `NotoSansCJKtc-Regular.ttf` | Traditional Chinese (TC) |
| `NotoSansCJKjp-Regular.ttf` | Japanese |
| `NotoSansCJKkr-Regular.ttf` | Korean |

```bash
# From repo root (needs fonttools + system NotoSansCJK-Regular.ttc)
./scripts/extract-cjk-fonts.sh
```

Loaded at GUI startup via `install_fonts` as soft-fail assets (`cjk_sc`, `cjk_tc`, `cjk_jp`, `cjk_kr`).

Other scripts (system fonts, not in this folder):

- Arabic / Urdu / Farsi: Harmattan, Mirza
- Hindi (Devanagari): Martel
- Cyrillic / Latin / Armenian: DejaVu Sans
