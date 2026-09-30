# Data

Nothing in `data/raw/` is committed.

## Course dataset (`track_f_student_success_{train,test}.csv`)

Supplied with the first-year Machine Learning course at PSL. Put the two CSVs in
`data/raw/`. Whether they may be redistributed is not settled, so they are not in the
repository. `pathfinder.data.validate` checks columns and value ranges on load.

## OULAD (for `experiments/05_early_warning_oulad.py`)

Open University Learning Analytics Dataset: Kuzilek, J., Hlosta, M. and Zdrahal, Z.
(2017), *Open University Learning Analytics dataset*, Scientific Data 4, 170171.
Licence CC BY 4.0. The Open University's own download page has moved; the copy used
here is the UCI Machine Learning Repository's
(https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset),
downloaded on 2026-09-30:

    open+university+learning+analytics+dataset.zip   46,748,244 bytes
    SHA-256 f2ed1902616c1fe8d2824d872c0b7d2d72be435bf0124d077044fe4be2c6d3e4

Its contents match the counts published with the dataset (22 module presentations,
32,593 student registrations, 10,655,280 rows of daily clicks);
`tests/test_oulad_published.py` checks this whenever the files are present. Unzip the
CSVs into `data/raw/oulad/`:

    studentInfo.csv  studentRegistration.csv  studentVle.csv
    assessments.csv  studentAssessment.csv

`pathfinder.oulad.load` refuses a missing file or column and reads `?` as missing.
