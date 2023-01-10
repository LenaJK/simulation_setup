#! /bin/bash

parentd=$PWD
for d in NPHrand_*; do
    cd $d
    if [[ -e output2.xyz ]]; then
	cat output.xyz output2.xyz > output3.xyz
	mv output3.xyz output.xyz
	rm output2.xyz
    fi
    cd $parentd
done
