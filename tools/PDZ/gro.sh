echo -e "1\n" | gmx trjconv -f conf481.gro -s topol.tpr -o md_nopbc.gro -ur compact -pbc mol
echo -e "19\n1\n" | gmx trjconv -f md_nopbc.gro -s topol.tpr -o md_cen.gro -center -n index.ndx 
echo -e "1\n" | gmx trjconv -f md_cen.gro -s topol.tpr -o md_res.gro -ur compact -pbc res
echo -e "1\n" | gmx trjconv -f md_res.gro -s topol.tpr -o md_done.gro -ur compact -pbc mol
rm -f md_nopbc.gro md_cen.gro md_res.gro
