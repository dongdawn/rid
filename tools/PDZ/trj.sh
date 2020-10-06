echo -e "1\n" | gmx trjconv -f traj_comp.xtc -s md-nosol.tpr -o md_nopbc.xtc -ur compact -pbc mol
echo -e "15\n1\n" | gmx trjconv -f md_nopbc.xtc -s md-nosol.tpr -o md_cen.xtc -center -n index_p.ndx
echo -e "1\n" | gmx trjconv -f md_cen.xtc -s md-nosol.tpr -o md_res.xtc -ur compact -pbc res
echo -e "1\n" | gmx trjconv -f md_res.xtc -s md-nosol.tpr -o md_done.xtc -ur compact -pbc mol
rm -f md_nopbc.xtc md_cen.xtc md_res.xtc
