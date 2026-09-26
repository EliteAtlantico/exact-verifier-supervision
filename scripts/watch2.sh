#!/usr/bin/env bash
# Emit one line per new finished cell (laptop logs) and per new remote commit (Khalil's push).
cd /c/Users/alexa/exact-verifier-supervision
seen1=$(grep -c "acc \|FAILED\|STOP\|DONE" results/laptop_queue4.log 2>/dev/null); seen3=$(grep -c "acc \|FAILED\|STOP\|DONE" results/laptop_queue3.log 2>/dev/null); seen1=${seen1:-0}; seen3=${seen3:-0}
lastfk=''; head0=$(git ls-remote origin refs/heads/main 2>/dev/null | cut -c1-7); t=0
while true; do
  for f in laptop_queue4 laptop_queue3; do
    n=$(grep -c "acc \|FAILED\|STOP\|DONE" results/$f.log 2>/dev/null); n=${n:-0}
    if [ "$f" = laptop_queue4 ]; then old=$seen1; else old=$seen3; fi
    if [ "$n" -gt "$old" ]; then grep "acc \|FAILED\|STOP\|DONE\|^run " results/$f.log | tail -$(( (n-old)*2 )) | tr '\n' ' ' | sed "s/^/[$f] /"; echo
      if [ "$f" = laptop_queue4 ]; then seen1=$n; else seen3=$n; fi; fi
  done
  t=$((t+1))
  if [ $((t % 10)) -eq 0 ]; then h=$(git ls-remote origin refs/heads/main 2>/dev/null | cut -c1-7); b=$(git ls-remote origin refs/heads/khalil-5090-results 2>/dev/null | cut -c1-7)
    if [ -n "$h" ] && [ "$h" != "$head0" ]; then echo "[remote] main moved to $h"; head0=$h; fi
    if [ -n "$b" ]; then echo "[remote] branch khalil-5090-results at $b"; fi
    fk=$(gh api repos/Libritor/exact-verifier-supervision/forks --jq '.[] | .full_name + "@" + .pushed_at' 2>/dev/null | tr '
' ' ')
    if [ -n "$fk" ] && [ "$fk" != "$lastfk" ]; then echo "[fork] $fk"; lastfk="$fk"; fi; fi
  sleep 30
done
