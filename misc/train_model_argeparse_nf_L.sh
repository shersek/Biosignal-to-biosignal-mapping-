for L in 2 4 6 
	do
	for n_f in 4 16 64 
		do  
			
			python3 train_model_argeparse_nf_L.py --n_f $n_f --L $L

		done
	done
