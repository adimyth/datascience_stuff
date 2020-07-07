if [ $# -eq 0 ];then
    for file in *.csv;
    do
        echo $file;
	    printf "# Rows: "; awk '{print NR}' $file| tail -n 1;
	    printf "# Columns: "; head -1 $file | sed 's/[^,]//g' | wc -c;
	    printf "\n\n"
    done;
else
    for file in $(readlink -f $1/*.csv);
    do
	    echo ${file##*/}
        printf "# Rows: "; awk '{print NR}' $file| tail -n 1;
	    printf "# Columns: "; head -1 $file | sed 's/[^,]//g' | wc -c;
	    printf "\n\n"
    done;
fi

