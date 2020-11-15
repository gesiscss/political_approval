#!/bin/bash

# reference: https://www.cyberciti.biz/faq/bash-for-loop/
mkdir temp_data
python data_prep.py
source deactivate
source activate tensorenv

for j in {0..4}
do 
	for i in Donald Emmanuel Jacob Recep Hillary Joko Vladimir
	do
		#mkdir out3/dssd/run_$i/run_$j
		python conditional.py $i $j
	done
done

rm -rf temp_data
source deactivate
source activate stancemaster
python data_combine.py