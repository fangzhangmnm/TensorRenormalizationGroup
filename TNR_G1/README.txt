The Python 3 source code provided here can be used reproduce the results
published in arXiv:1512.03846, the article that this code accompanies. The
exact commands that one needs to run to produce the plots shown in the article
are listed below.

Many parts of the code provide additional features that are not necessary for
producing the results published in the article. Many of these features are
functional, but many may also be in a broken state, as the code here represents
a snapshot of a code base that is in constant development. Some documentation
is provided, most of which is up-to-date, but some of which may not be.

Note also that the implementation of tensor network renormalization provided
here lacks many of the improvements introduced in the appendices of
arXiv:1509.07484. 

Because of these reasons the code presented here should not be considered a
reference implementation of the algorithms discussed in the article. Rather
it should be viewed as the ultimate answer to questions like "what exactly did
these people do in part X of their algorithm?", as well as proof that the
techniques discussed in the paper produce the claimed results.

However, you are free to use the code as you see fit, as long as you abide to
the (very relaxed) license terms of the MIT license, described in the file
LICENSE.

If you have any questions about the code, please contact Markus Hauru at
markus@mhauru.org.

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

The following are the commands one needs to run to reproduce the results we
present in arXiv:1512.03846. However, to get started, we recommend running
these commands with lower bond dimensions (parameters chis_tnr and chis_trg)
and for the exact diagonalization results with smaller system sizes (parameter
block_width). The results you'll obtain that way will be slightly less
accurate, but the runs will finish much faster. Some of the commands listed
below will require dozens of gigs of RAM and several days of running even on a
powerful multicore machine.

TNR:

Ising without defects:
python3 scaldimer.py -model 'ising' -algorithm 'TNR' -initial4x4 True -symmetry_tensors True -horz_refl True -fix_gauges True -reuse_initial True -return_pieces True -return_gauges True -do_momenta True -do_coarse_momenta True -plot_by_qnum False -chis_tnr 14 -chis_trg 28 -iter_count 5 -block_width 4 -n_dims_do 30 -n_dims_plot 30 -max_dim_plot 4.5 -plot_by_momenta True

Ising with D_epsilon:
python3 scaldimer.py -model 'ising' -algorithm 'TNR' -initial4x4 True -symmetry_tensors True -horz_refl True -fix_gauges True -reuse_initial True -return_pieces True -return_gauges True -do_momenta True -do_coarse_momenta True -plot_by_qnum False -chis_tnr 14 -chis_trg 28 -iter_count 5 -block_width 4 -n_dims_do 50 -defect_angles '0, 3.141592653589793' -n_dims_plot 50 -max_dim 4.3 -plot_by_momenta True -draw_defect_angle False -xtick_rotation "0,45"

Ising with D_sigma:
python3 KW_scaldims.py -symmetry_tensors True -reuse_initial True -return_pieces True -do_momenta True -do_coarse_momenta True -plot_by_qnum False -n_discard 2 -chis_tnr 11 -chis_trg 22 -iter_count 5 -block_width 4 -n_dims_do 40 -n_dims_plot 40 -plot_by_momenta True -max_dim_plot 4.9 -xtick_rotation 45

3-State Potts with symmetry defects:
python3 scaldimer.py -model 'potts3' -algorithm 'TNR' -initial2x2 True -symmetry_tensors True -horz_refl True -fix_gauges True -reuse_initial True -return_pieces True -return_gauges True -do_momenta True -do_coarse_momenta True -plot_by_qnum False -chis_tnr 15 -chis_trg 30 -iter_count 5 -block_width 4 -qnums_do '0,1,2' -n_dims_do 60 -defect_angles '0,2.0943951023931953,4.1887902047863905' -n_dims_plot 60 -max_dim_plot 4.4 -plot_by_momenta True -draw_defect_angle False

Continuous family of defects for the Ising model:
python3 cdf_scaldimer.py -reuse_initial True -return_pieces True -plot_by_qnum False -n_discard 2 -chis_tnr 11 -chis_trg 22 -iter_count 5 -block_width 4 -n_dims_do 50 -n_dims_plot 50 -max_dim_plot 8.9 -gs '0.0,1.0,11' -symmetry_tensors True -draw_exact_circles False



Exact diagonalization:

Ising without defects:
python3 ed_scaldimer.py -model ising -n_dims_do 50 -block_width 18 -symmetry_tensors False -do_momenta True -n_dims_plot 50 -max_dim_plot 4.5 -plot_by_qnum False -plot_by_momenta True -sep_qnums True -do_eigenvectors True

Ising with D_epsilon:
python3 ed_scaldimer.py -model ising -n_dims_do 70 -block_width 18 -do_momenta True -n_dims_plot 70 -max_dim_plot 4.4 -plot_by_qnum False -plot_by_momenta True -draw_defect_angle False -symmetry_tensors False -defect_angles '0,3.141592653589793' -xtick_rotation 45 -sep_qnums True -do_eigenvectors True

Ising with D_sigma:
python3 ed_scaldimer.py -model ising -n_dims_do 70 -block_width 18 -do_momenta True -n_dims_plot 70 -max_dim_plot 5.0 -plot_by_qnum False -plot_by_momenta True -draw_defect_angle False -KW True -symmetry_tensors False -xtick_rotation 45 -sep_qnums True -do_eigenvectors True

Continuous family of defects:
python3 cdf_ed_scaldimer.py -model ising -n_dims_do 60 -n_dims_plot 60 -symmetry_tensors False -gs '0,1,11' -block_width 18 -draw_exact_circles False -sep_qnums True -do_eigenvectors True

