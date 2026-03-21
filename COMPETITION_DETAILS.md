# BirdCLEF+ 2026 - Full Competition Details

Source: https://www.kaggle.com/competitions/birdclef-2026

## Overview

- **Title:** BirdCLEF+ 2026
- **Subtitle:** Acoustic Species Identification in the Pantanal, South America
- **Host:** Cornell Lab of Ornithology
- **Sponsor:** Google Research & Cornell Lab of Ornithology
- **Type:** Research Code Competition
- **Total Prizes:** $50,000
- **Data License:** CC BY-NC-SA 4.0
- **Participation (as of March 2026):** ~6,183 entrants, ~997 participants, ~934 teams, ~8,511 submissions

## Task Description

Identify which species (birds, amphibians, mammals, reptiles, insects) are calling in
audio recordings made in the Brazilian Pantanal. This is a **multi-label classification**
problem: for each 5-second audio segment, predict the probability that each of 234 species
is present.

The Pantanal is a wetland spanning 150,000+ km^2 across Brazil and neighboring countries,
home to over 650 bird species plus countless other animals. A network of 1,000 acoustic
recorders is being deployed across the Pantanal for passive acoustic monitoring (PAM).

Key challenge: Models must work across different habitats, withstand messy field-collected
data, and support evidence-based conservation decisions.

## Timeline

| Date | Event |
|------|-------|
| March 11, 2026 | Start Date |
| May 27, 2026 | Entry Deadline (must accept rules before this date) |
| May 27, 2026 | Team Merger Deadline (last day to join/merge teams) |
| June 3, 2026 | Final Submission Deadline |

All deadlines are at 11:59 PM UTC. Organizers reserve the right to update the timeline.

### Working Note Dates (optional)
| Date | Event |
|------|-------|
| June 3, 2026 | Competition deadline |
| June 17, 2026 | Working note submission deadline |
| June 24, 2026 | Notification of acceptance |
| July 6, 2026 | Camera-ready submission deadline |

## Evaluation Metric

**Macro-averaged ROC-AUC** that skips classes with no true positive labels in the test set.

This means:
- ROC-AUC is computed per species
- Species that do not appear in the test data are excluded from the average
- The final score is the mean of per-species ROC-AUC values (only for species present in test)

## Submission Format

- For each `row_id`, predict the probability that each of 234 species is present
- One column per species (234 species columns)
- Each row covers a **5-second window of audio**
- `row_id` format: `[soundscape_filename]_[end_time]`
  - Example: segment 00:15-00:20 of `BC2026_Test_0001_S05_20250227_010002.ogg` has row_id `BC2026_Test_0001_S05_20250227_010002_20`
- Submission file must be named `submission.csv`

## Data Description

**Total size:** ~16.14 GB, 46,213 files (ogg, csv, txt)

### Files

#### train_audio/
- Short recordings of individual bird, amphibian, reptile, mammal, and insect sounds
- Sources: xeno-canto.org and iNaturalist
- Resampled to **32 kHz**, converted to **ogg format**
- Filenames: `[collection][file_id_in_collection].ogg`

#### test_soundscapes/
- Populated at submission time (hidden test set)
- ~600 recordings, each **1 minute long**, ogg format, **32 kHz**
- Filename format: `BC2026_Test_<fileID>_<site>_<date>_<timeUTC>.ogg`
  - Example: `BC2026_Test_0001_S05_20250227_010002.ogg` = file ID 0001, site S05, Feb 27 2025, 01:00 UTC
- Loading all test soundscapes should take ~5 minutes
- **Not all training species appear in the test data**

#### train_soundscapes/
- Additional audio from roughly the same recording locations as test_soundscapes
- Same naming convention as test_soundscapes
- Some recording sites overlap between train and test, but **dates and times do NOT overlap**
- Some have been labeled by expert annotators (see train_soundscapes_labels.csv)
- **Important:** Some species in the hidden test data may ONLY have train samples in labeled train_soundscapes (not in train_audio from XC/iNat)
- Not all species from train_soundscapes appear in test_soundscapes

#### train_soundscapes_labels.csv
- Ground truth for a subset of train_soundscapes
- Columns:
  - `filename`: soundscape filename
  - `start`: start time of 5-second segment
  - `end`: end time of 5-second segment
  - `primary_label`: semicolon-separated list of species codes present in this segment

#### train.csv
Key metadata fields:
- `primary_label`: Species code (eBird code for birds, iNaturalist taxon ID for non-birds)
- `secondary_labels`: Other species also in the recording (may be incomplete)
- `latitude` & `longitude`: Recording coordinates (useful for geographic diversity / call dialects)
- `author`: Recording uploader
- `filename`: Associated audio file name
- `rating`: Quality rating 1-5 (0.5 reduction for background species; 0 = no rating; iNat has no ratings)
- `collection`: Either "XC" (Xeno-canto) or "iNat" (iNaturalist)

#### taxonomy.csv
- 234 rows = 234 class columns in submission
- Includes iNaturalist taxon ID and class name (Aves, Amphibia, Mammalia, Insecta, Reptilia)
- `primary_label` specifies the submission column name
- Some insect species are **sonotypes** (e.g., `47158son16` = insect sonotype 16) -- treated as classes despite lacking species-level ID; some occur in test data

#### sample_submission.csv
- Valid sample submission with correct format
- 234 species ID columns + row_id column

#### recording_location.txt
- Location info: Pantanal, Mato Grosso do Sul, Brazil, South America
- Recorder deployment site coordinates: Lat -16.5 to -21.6, Lon -55.9 to -57.6

## Code Requirements (Critical)

**This is a Code Competition.** Submissions must be made through Kaggle Notebooks.

| Constraint | Value |
|------------|-------|
| Runtime | CPU Notebook <= 90 minutes |
| GPU | Disabled (GPU submissions get only 1 minute runtime) |
| Internet | Disabled |
| External data | Freely & publicly available external data allowed, including pre-trained models |
| Output file | Must be named `submission.csv` |

## Competition Rules (Key Points)

- **Max team size:** 5
- **Max submissions per day:** 5
- **Final submissions for judging:** 2 (you select which ones)
- **External data/models:** Allowed if publicly available and free for all participants
- **Data license:** CC BY-NC-SA 4.0 (non-commercial use only)
- **No multiple accounts**

## Prizes

| Place | Prize |
|-------|-------|
| 1st | $15,000 |
| 2nd | $10,000 |
| 3rd | $8,000 |
| 4th | $7,000 |
| 5th | $5,000 |
| Best Working Note (x2) | $2,500 each |

## Key Strategic Notes for Solution Planning

1. **Multi-species, multi-taxa:** Not just birds -- includes amphibians, mammals, reptiles, insects (234 classes total)
2. **Soundscape inference:** Test data is 1-minute continuous recordings split into 5-second segments
3. **Domain gap:** Training data is mostly clean individual recordings (XC/iNat), but inference is on noisy field recordings
4. **Hidden species:** Some test species may only appear in train_soundscapes labels, not in train_audio
5. **No GPU at inference:** Must run on CPU in 90 minutes for ~600 one-minute files
6. **Metric skips absent classes:** Species not in test set don't affect score -- no penalty for false positives on absent species
7. **Audio specs:** All audio at 32 kHz, ogg format
8. **Geographic info available:** Lat/lon and site IDs could be useful features
9. **Insect sonotypes:** Some classes are unidentified insect sound types, not species
10. **Train soundscape labels use semicolons:** Multiple species per segment separated by semicolons
