# Contributing to RFFPLA

## Commit Message Convention

Every commit must reference a Jira issue key:

RFFPLA-XX: short description in imperative mood

**Examples:**

RFFPLA-46: extract preprocess.py from app.py
RFFPLA-47: add README and CONTRIBUTING docs
RFFPLA-48: add architecture diagram and decision log

**Rules:**
- Start with the Jira issue key
- Use imperative mood: "add", "fix", "update" not "added" or "adding"
- Keep the description under 72 characters
- No full stops at the end

## Branch Naming

feature/RFFPLA-XX-short-description
fix/RFFPLA-XX-short-description

## File Placement

| Type                | Location                                             |
|---                  |---                                                   |
| Dashboard code      | `app.py`                                             |
| Signal processing   | `preprocess.py`                                      |
| Shared constants    | `config.py` — never hardcode signal params elsewhere |
| Trained models      | `models/` — never commit files >100MB                |
| Raw recordings      | `data/` — never commit `.c64` files                  |
| Documentation       | `docs/`                                              |
| Evaluation outputs  | `results/`                                           |

## What Not to Commit

See `.gitignore`. Never commit:
- `.c64` raw recording files
- `.npy` dataset arrays
- `.h5` model files
- `__pycache__/`
- `.env` files

