# Data Folder

Raw recording files and processed arrays are stored here locally
but are NOT committed to GitHub (see .gitignore).

## Structure
data/
├── raw/
│   ├── auth_session1/    ← .c64 files from authorized device session 1
│   ├── auth_session2/    ← .c64 files from authorized device session 2
│   └── rogue_session1/   ← .c64 files from rogue device session 1
└── processed/
├── s1/               ← .npy arrays output from preprocess.py
└── r1/               ← .npy arrays from rogue session

## File sizes

- Raw .c64 files: ~960 MB per 60-second recording
- Processed .npy files: ~400 KB per session

## Access

Raw recordings and processed arrays are stored in Google Drive:
MyDrive/RFFLPA/

Contact the project team for access.

