for i in 'x' 'y' 'z' 'xy' 'xyz' 
	do
	python3 train_model_argeparse_a000.py --n_f 16 --L 6 --axis $i
	done
