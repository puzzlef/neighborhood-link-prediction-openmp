#!/usr/bin/env bash
src="neighborhood-link-prediction-openmp"
out="$HOME/Logs/$src$1.log"
ulimit -s unlimited
printf "" > "$out"

# Download program
if [[ "$DOWNLOAD" != "0" ]]; then
  rm -rf $src
  git clone https://github.com/puzzlef/$src
  cd $src
  git checkout approach-parallel-sort
fi

# Fixed config
: "${TYPE:=float}"
: "${MAX_THREADS:=32}"
: "${REPEAT_BATCH:=1}"
: "${REPEAT_METHOD:=1}"
# Parameter sweep for batch (randomly generated)
: "${BATCH_UNIT:=%}"
: "${BATCH_LENGTH:=1}"
: "${BATCH_DELETIONS_BEGIN:=0.0000001}"
: "${BATCH_DELETIONS_END:=0.1}"
: "${BATCH_DELETIONS_STEP:=*=10}"
: "${BATCH_INSERTIONS_BEGIN:=0}"
: "${BATCH_INSERTIONS_END:=0}"
: "${BATCH_INSERTIONS_STEP:=+=10}"
# Define macros (dont forget to add here)
DEFINES=(""
"-DTYPE=$TYPE"
"-DMAX_THREADS=$MAX_THREADS"
"-DREPEAT_BATCH=$REPEAT_BATCH"
"-DREPEAT_METHOD=$REPEAT_METHOD"
"-DBATCH_UNIT=\"$BATCH_UNIT\""
"-DBATCH_LENGTH=$BATCH_LENGTH"
"-DBATCH_DELETIONS_BEGIN=$BATCH_DELETIONS_BEGIN"
"-DBATCH_DELETIONS_END=$BATCH_DELETIONS_END"
"-DBATCH_DELETIONS_STEP=$BATCH_DELETIONS_STEP"
"-DBATCH_INSERTIONS_BEGIN=$BATCH_INSERTIONS_BEGIN"
"-DBATCH_INSERTIONS_END=$BATCH_INSERTIONS_END"
"-DBATCH_INSERTIONS_STEP=$BATCH_INSERTIONS_STEP"
)

# Run
g++ ${DEFINES[*]} -std=c++17 -O3 -fopenmp main.cxx
stdbuf --output=L ./a.out ~/Data/web-Stanford.mtx      0 0 2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/Data/web-BerkStan.mtx      0 0 2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/Data/web-Google.mtx        0 0 2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/Data/web-NotreDame.mtx     0 0 2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/Data/soc-Slashdot0811.mtx  0 0 2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/Data/soc-Slashdot0902.mtx  0 0 2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/Data/soc-Epinions1.mtx     0 0 2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/Data/coAuthorsDBLP.mtx     1 0 2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/Data/coAuthorsCiteseer.mtx 1 0 2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/Data/soc-LiveJournal1.mtx  0 0 2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/Data/coPapersCiteseer.mtx  1 0 2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/Data/coPapersDBLP.mtx      1 0 2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/Data/indochina-2004.mtx    0 0 2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/Data/italy_osm.mtx         1 0 2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/Data/great-britain_osm.mtx 1 0 2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/Data/germany_osm.mtx       1 0 2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/Data/asia_osm.mtx          1 0 2>&1 | tee -a "$out"

# Signal completion
curl -X POST "https://maker.ifttt.com/trigger/puzzlef/with/key/${IFTTT_KEY}?value1=$src$1"
