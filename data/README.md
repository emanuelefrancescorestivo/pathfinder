# Data

Nothing in `data/raw/` is committed.

## Course dataset (`track_f_student_success_{train,test}.csv`)

Supplied with the first-year Machine Learning course at PSL. Put the two CSVs in
`data/raw/`. Whether they may be redistributed is not settled, so they are not in the
repository. `pathfinder.data.validate` checks columns and value ranges on load.

## OULAD (for `experiments/05_early_warning_oulad.py`)

Open University Learning Analytics Dataset: Kuzilek, J., Hlosta, M. and Zdrahal, Z.
(2017), *Open University Learning Analytics dataset*, Scientific Data 4, 170171.
Download it from the Open University (https://analyse.kmi.open.ac.uk/open_dataset) or
the UCI Machine Learning Repository. Check the licence on the page you download from,
then unzip the CSVs into `data/raw/oulad/`:

    studentInfo.csv  studentRegistration.csv  studentVle.csv
    assessments.csv  studentAssessment.csv

`pathfinder.oulad.load` refuses a missing file or column and reads `?` as missing.
