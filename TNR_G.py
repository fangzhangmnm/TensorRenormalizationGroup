import numpy as np
import warnings
from scon import scon


""" Library for doing TNR on a square lattice. Written in python3. """


# When tensors are represented diagramatically, the numbering of indices
# is thought to run clock-wise, starting from 9 o'clock. If several
# indices are pointing in the same direction the order within them is
# bottom-up and left to right. This of course makes no mathematical
# difference, but may help with reading the code. This ordering may not
# always hold inside functions between intermediate steps (hopefully I
# have commented on all such exceptions), but is used at least when
# returning values.

default_pars = {
    "chis_tnr": None,
    "chis_trg": None,
    "horz_refl": False,
    "opt_eps_chi": 0,
    "opt_iters_tens": 1,
    "opt_eps_conv": 1e-9,
    "opt_max_iter": np.inf,
    "A_chis": None,
    "A_eps": 0,
    "fix_gauges": False,
    "reuse_initial": False,
    "print_errors": 0,
    "return_pieces": False,
    "return_gauges": False
    }

default_gauges = {
    "G_hh": None,
    "G_hv": None,
    "G_vh": None,
    }

default_pieces = {
    "A_part_list": None,
    "u": None,
    "v": None,
    "w": None,
    "B": None,
    "BUS": None,
    "BSV": None,
    "z": None,
    }


def tnr_step(A_list, log_fact=0, pars=dict(), gauges=dict(), pieces=dict(),
             **kwargs):

    """ A single TNR step.
    
    Parameters:
    - A_list is a list of 2 tensors that form the block which is to be
      coarse-grained. It can also be a single tensor, in which case 2
      copies of it are used. The block to be coarse-grained is formed
      like this:
             |               | 
      ----A_list[0]------A_list[1]---
             |               |        .
      --A_list[0]^dg---A_list[1]^dg--
             |               |
      ^dg (dagger) denotes up-down transposing and complex conjugating.
    - chis_tnr is either an integer or an iterable of integers. These
      are the bond dimensions for truncation in the first (TNR) step of
      algorithm.  The smallest one of them (assuming there are several)
      is used such that the truncation error is smaller than
      ops_eps_chi.
    - chis_trg is the same as chis_tnr, but for the second (TRG-like)
      step of the algorithm.
    - horz_refl specifies whether we can assume reflection symmetry in
      the horizontal direction in the following sense: If horz_refl =
      True the element A_list[1] is ignored and in stead is assumed to
      be the conjugate of A_list[0], where conjugation here means
      left-right reflection and complex conjugation. This also
      guarantees that the resulting A_new is left-right reflection +
      complex conjugation symmetric up to gauge factors. Defaults to
      False. See also G_hh and return_gauges.
    - G_hh is one of the gauge matrices of the given A. See
      return_gauges for explanations of what the gauge matrices are.
      G_hh is only needed if horz_refl=True, and the other gauge
      matrices are not needed at all for tnr_step. Defaults to the
      identity matrix.
    - opt_eps_chi is the limit for the truncation error when choosing
      the dimension to truncate to. For details see chis_tnr and
      chis_trg. Defaults to None, in which case the largest allowed bond
      dimension is always used.
    - opt_iters_tens is the number of iterations done when optimizing a
      single tensor, before moving on to the next tensor. This is only
      relevant for optimizations where the cost function is linearized
      and a single iteration does not result in the optimal solution for
      the given (non-linearized) environment. Defaults to 1.
    - opt_eps_conv (conv for convergence) is the limit below which the
      change in the cost function for successive iterations of the
      optimization algorithm must fall for the loop to finish. Defaults
      to 1e-9.
    - opt_max_iter is the maximum number of iterations done in the
      optimization of the tensors for every chi. This is only relevant
      if convergence is never reached as in the case ops_eps_conv = 0.
      Defaults to np.inf.
    - A_eps is the maximum allowed truncation error when SVDing the
      tensors in A_list. This SVD is done simply to boost performance,
      and the error at this step should be kept small.  Defaults to 0.
    - A_chis is the possible truncation dimensions when SVDing A, see
      A_eps. Defaults to None which is interpreted as "use any dimension
      necessary".
    - fix_gauges determines whether to fix the gauge freedom of SVD when
      building z and splitting B to be the same as at the previous step.
      If fix_gauges=True, pieces needs to be given. Defaults to False.
    - log_fact is the logarithm of the factor by which the original A
      should be multiplied to get the physical tensor. Defaults to 0.
    - The higher the print_errors number is, the more intermediate steps
      will produce text output that tells about the progression of the
      algorithm and the errors induced in the coarse-graning. Note that
      asking for more error output induces non-trivial computations that
      slow down the function.
    - return_pieces determines whether the tensors used in building up
      the coarse-grained A_new are also returned. Defaults to False.
    - return_gauges determines whether to also return the gauge matrices
      related to different conjugate transposes of the final tensor.  If
      return_gauges = True then the last value returned is the tuple
      (G_hh, G_hv, G_vh), where the different Gs are matrices such that

            |                     |
      |--   |                    G_hv
      |     |                     |
      ---A_new^*---  =  --G_hh--A_new--G_hh--
            |     |               |
            |   --|              G_hv^dg
            |                     |

      and

           |--                    |
           | |                    |  
           |                      |
      ---A_new^*---  =  --G_vh--A_new--G_vh^dg--
           |                      |
         | |                      |  
         --|                      |

      Here ^* denotes complex conjugation and the bending of the legs
      denotes different transposes. Here G_hh is a diagonal matrix of
      +/-1, and the others are unitary.  The logic in the naming of the
      gauge matrices is that h refers to horizontal and v refers to
      vertical and the first letter in the subscript tells which
      transpose of A_new this matrix is related to and the second one
      tells which legs (horizontal or vertical) of A_new this matrix
      should be attached to.

    Returns: A_new, new_log_fact, pieces, (G_hh, G_hv, G_vh),
    with the last two being opitional.
    - tnr_step returns the coarse-grained tensor A_new and the logarithm
      new_log_fact of the related factor such that A_new *
      exp(new_log_fact) is the physical tensor.
    - If return_pieces=True then a third value is also returned that is
      a dictionary of the tensors A_part_list, u, v, w, B, BUS, BSV and
      z, with these names as string keys.
    - If return_gauges=True then the gauge matrices G_hh, G_hv and G_vh
      that map A_new to its various conjugates are also returned as a
      dict with the above names as string keys.
    """

    # Format the parameters and initialize A_part_list with the SVDs of
    # the original As.
    A_list, pars, gauges, pieces = format_parameters(A_list, default_pars,
                                                     default_gauges,
                                                     default_pieces, pars,
                                                     gauges, pieces, **kwargs)
    if pars["print_errors"] > 0:
        print("Beginning a TNR step.")
    A_list = symmetrize_A_list(A_list, pars, gauges)
    A_part_list = split_A_list(A_list, pars)

    # Determine whether to use the SVD of the original tensor, based on
    # a rought estimate of whether it's computationally advantageous.
    A_NW = A_part_list[0][0]
    chi_orig = type(A_NW).flatten_dim(A_NW.shape[0])
    chi_split = type(A_NW).flatten_dim(A_NW.shape[2])
    pars["use_parts"] = chi_split < chi_orig**(3/2)
    if pars["print_errors"] > 1:
        if pars["use_parts"]:
            print("Using the SVDed A.")
        else:
            print("Not using the SVDed A.")

    # Obtain the optimized u, v, w.
    u, v, w = build_uvw(A_list, A_part_list, pars, gauges, pieces)

    # Build the intermediate tensors B and z and use them to put
    # together A_new.
    B = build_B(u, v, w, A_list, A_part_list, pars)
    split_B_result = split_B(B, pars)
    BUS, BSV = split_B_result[0:2]
    if pars["return_gauges"]:
        G_hh = split_B_result[2]
    z = build_z(v, w, BUS, BSV, pars)
    A_new = build_A_new(v, w, z, BUS, BSV)

    # Scale A_new to have largest values around unity and use that to
    # update new_log_fact.
    fact = A_new.abs().max()
    if fact != 0:
        A_new = A_new/fact
        new_log_fact = np.log(fact) + 4*log_fact
    else:
        new_log_fact = 0

    if pars["fix_gauges"] and A_new.shape == A_list[0].shape:
        if pars["return_gauges"] and pars["horz_refl"]:
            A_new, BUS, BSV, z, G_hh = fix_A_new_gauge(A_new, A_list[0], pars, BUS,
                                                       BSV, z, G_hh=G_hh)
        else:
            A_new, BUS, BSV, z = fix_A_new_gauge(A_new, A_list[0], pars, BUS,
                                                 BSV, z)

    # Print error in trace of the block, if asked for.
    if pars["print_errors"] > 2:
        err = print_Z_error(A_list, log_fact, A_new, new_log_fact)

    # Put together the values to be returned and return.
    return_value = (A_new, new_log_fact)
    if pars["return_pieces"]:
        pieces = {'A_part_list': A_part_list, 'u': u, 'v': v, 'w': w,
                      'B': B, 'BUS': BUS, 'BSV': BSV, 'z': z} 
        return_value = return_value + (pieces,)
    if pars["return_gauges"]:
        if pars["horz_refl"]:
            G_hv = optimize_G_hv(A_new, G_hh, pars)
        else:
            G_hv = None
        G_vh = optimize_G_vh(A_new, pars)
        gauges = {'G_hh': G_hh, 'G_hv': G_hv, 'G_vh': G_vh}
        return_value = return_value + (gauges,)

    return return_value


# Functions for initializing the parameters.


def format_parameters(A_list, default_pars, default_gauges, default_pieces,
                      pars, gauges, pieces, **kwargs):
    """ Formats some of the parameters given to tnr_step to a canonical
    form.
    """
    # Make sure A_list is a list of 2 tensors.
    if type(A_list) == list or type(A_list) == tuple:
        A_list = list(A_list)
    else:
        A = A_list
        A_list = [A, A]

    # Create pars, gauges and pieces.
    # Values are taken primarily from kwargs, then from pars, gauges and
    # pieces, and finally from default. Only ones listed in defaults are
    # used, others are ignored.
    new_pars = default_pars.copy()
    new_pars.update(pars)
    for k in default_pars:
        if k in kwargs:
            new_pars[k] = kwargs[k]
    pars = new_pars

    new_gauges = default_gauges.copy()
    new_gauges.update(gauges)
    for k in default_gauges:
        if k in kwargs:
            new_gauges[k] = kwargs[k]
    gauges = new_gauges

    new_pieces = default_pieces.copy()
    new_pieces.update(pieces)
    for k in default_pieces:
        if k in kwargs:
            new_pieces[k] = kwargs[k]
    pieces = new_pieces

    # Make sure chis_tnr and chis_trg are a lists of integers (or at
    # least singlet lists of one integer) and sorted from small to
    # large.
    if type(pars["chis_tnr"]) == int:
        pars["chis_tnr"] = [pars["chis_tnr"]]
    else:
        pars["chis_tnr"] = list(pars["chis_tnr"])
    pars["chis_tnr"] = sorted(pars["chis_tnr"])
    if type(pars["chis_trg"]) == int:
        pars["chis_trg"] = [pars["chis_trg"]]
    else:
        pars["chis_trg"] = list(pars["chis_trg"])
    pars["chis_trg"] = sorted(pars["chis_trg"])

    # If several chis to loop over are given but there is no epsilon to
    # determine sufficient accuracy, then just use the largest chi.
    if pars["opt_eps_chi"] == 0:
        pars["chis_tnr"] = [max(pars["chis_tnr"])]
        pars["chis_trg"] = [max(pars["chis_trg"])]
        pars["opt_eps_chi"] = np.inf

    # If some parameters don't make sense, raise warnings.
    if pars["horz_refl"] and not pars["return_gauges"]:
        warnings.warn("In TNR, horz_refl is True but return_gauges is False")
    if pars["reuse_initial"] and not pars["return_pieces"]:
        warnings.warn("In TNR, reuse_initial is True but return_pieces is False")
    if pars["fix_gauges"] and not pars["return_pieces"]:
        warnings.warn("In TNR, fix_gauges is True but return_pieces is False")

    return A_list, pars, gauges, pieces


def symmetrize_A_list(A_list, pars, gauges):
    """ Symmetrizes A_list according to the value of horz_refl. """
    new_list = A_list.copy()
    if pars["horz_refl"]:
        new_list[1] = new_list[0].conjugate().transpose((2,1,0,3))
        if gauges["G_hh"] is not None:
            new_list[1] = scon((new_list[1], gauges["G_hh"], gauges["G_hh"]),
                               ([1,-2,3,-4], [-1,1], [3,-3]))
    else:
        if A_list[1].dirs == A_list[0].dirs:
            new_list[1] = A_list[1].flip_dir(1)
            new_list[1] = new_list[1].flip_dir(3)
    return new_list


def split_A_list(A_list, pars):
    """ SVDs the tensors in A_list and returns a list of lists with the
    parts in them. See split_A for details.
    """
    def split_helper(A, direction, i):
        return split_A(A, eps=pars["A_eps"], chis=pars["A_chis"],
                       direction=direction,
                       print_errors=pars["print_errors"]-i)
    A_part_list = [split_helper(A_list[0], 'nwse', 0),
                   split_helper(A_list[1], 'nesw', 2)]
    return A_part_list


def split_A(A, eps=0, chis=None, direction='both', print_errors=0):
    """ Splits an A tensor by SVD into two. eps is the tolerance in the
    SVD, chis is the list of bond dimensions to try when splitting (by
    default the full possible range is tried) and direction specifies
    the way to split.  If direction == 'nesw' (case and space
    insensitive) the splitting is done so that the northeast and
    southwest components are created, and similarly for direction ==
    'nwse'. For all other values of direction both splittings are done.

    The values are returned in the order A_NW, A_SE, A_NE, A_SW,
    possibly leaving out some based on direction.
    """
    if print_errors>0:
        print('-Splitting A, with eps=%.3e.' % eps)
    ret_val = ()
    if direction.lower().replace(' ','') != 'nesw':
        A_NW, S, A_SE = A.split((0,1), (2,3), eps=eps, chis=chis,
                                    print_errors=print_errors,
                                    return_sings=True) 
        ret_val += (A_NW, A_SE)
        if(print_errors>0):
            print('Split A NWSE with chi=%i' % len(S))
    if direction.lower().replace(' ','') != 'nwse':
        A_SW, S, A_NE = A.split((0,3), (1,2), eps=eps, chis=chis,
                              print_errors=print_errors,
                              return_sings=True) 
        A_SW = A_SW.transpose((0,2,1))
        A_NE = A_NE.transpose((1,2,0))
        ret_val += (A_NE, A_SW)
        if(print_errors>0):
            print('Split A NESW with chi=%i' % len(S))
    return ret_val


# Functions for executing the first step of TNR: Optimizing u, v, and w. 


def build_uvw(A_list, A_part_list, pars, gauges, pieces):
    """ Produces the optimized isometries v and w and the unitary u. """
    if pars["print_errors"] > 0:
        print('-Optimizing u, v and w.')

    if pars["print_errors"] > 0 or len(pars["chis_tnr"]) > 1:
        # We will calculate the TNR error at some point, so we
        # precompute this outside the loop.
        orig_norm = np.sqrt(A4_frob_norm_sq(A_list, A_part_list, pars))
    # Loop over growing truncation dimensions until the truncation
    # error is small enough.
    for chi_num, chi in enumerate(pars["chis_tnr"]):
        if pars["print_errors"] > 2:
            print('Optimizing for chi = %i.' % chi)
        # Use the u from the previous chi as the starting point of
        # the optimization.
        u_init, v, w = initial_uvw(A_list, chi, pars, gauges, pieces)
        if chi == pars["chis_tnr"][0] or pars["reuse_initial"]:
            u = u_init
        else:
            del(u_init)
        new_cost = np.inf
        cost_change = np.inf
        # Keep optimizing u, v, w iteratively to maximitze the norm of
        # B until the relative change is less than opt_eps_conv.
        # If convergence is not reached in opt_max_iter steps, stop and
        # move on.
        counter = 0
        w_flat_shape = type(w).flatten_shape(w.shape)
        full_dim = w_flat_shape[1]*w_flat_shape[2]
        if full_dim > chi:
            # There is truncation, so optimization is necessary.
            while (cost_change > pars["opt_eps_conv"]
                   and counter < pars["opt_max_iter"]):
                counter += 1
                old_cost = new_cost
                v, w = optimize_isometries(u, v, w, A_list, A_part_list, pars,
                                           gauges)
                u, B_norm = optimize_disentangler(u, v, w, A_list, A_part_list,
                                                  pars, return_B_norm=True)
                # The cost_function is not needed because
                # optimize_disentangler gives us B_norm.
                new_cost = B_norm
                cost_change = np.abs((old_cost - new_cost)/new_cost)
                if pars["print_errors"] > 2:
                    print('\r  Change in norm when optimizing: %.3e'
                          % cost_change, end='')
        if pars["print_errors"] > 2:
            print('')  #To end the line of the running counter.
            if counter >= pars["opt_max_iter"]:
                print('  Maximum number of optimization iterations '
                      '(%i) reached.' % pars["opt_max_iter"])
            else:
                print('  Optimization converged after %i iterations.'
                      % counter)
        if pars["print_errors"] > 0 or chi_num < len(pars["chis_tnr"]) -1:
            # Compute the truncation error and move on if it's small enough.
            # Otherwise go to the next chi in chis and reoptimize.
            if counter == 0:
                B_norm = build_B(u, v, w, A_list, A_part_list, pars).norm()
            err = compute_TNR_error(u, v, w, A_list, A_part_list, pars,
                                    B_norm=B_norm, orig_norm=orig_norm)
            if err < pars["opt_eps_chi"]:
                break

    # Error printing, as asked for.
    if pars["print_errors"] > 0:
        print('Truncated bond dimension in TNR: %i' % chi)
        if 2 >= pars["print_errors"] > 1:
            if counter >= pars["opt_max_iter"]:
                print('Maximum number of optimization iterations '
                      '(%i) reached.' % pars["opt_max_iter"])
            else:
                print('Optimization converged after %i iterations.' % counter)
        print('Relative error in optimized, truncated TNR block: %.3e + %gj'
              %(np.real(err), np.imag(err)))
    return u, v, w


# At the moment this function is not in use.
def cost_function(u, v, w, A_list, A_part_list, pars):
    """ This is the cost function that is maximized in the optimization
    routine. However, note that changing this will not change the
    optimization step, this only determines when optimization is
    stopped.
    """
    upb = upper_block(u, v, w, A_list, A_part_list, pars)
    B = build_B(u, v, w, A_list, A_part_list, pars, upb=upb)
    norm = B.norm()
    return norm, upb


def initial_uvw(A_list, chi, pars, gauges, pieces):
    """ Returns the initial disentangler and isometries that are the
    starting point of the optimization. The initial u is the identity,
    the initial isometries are SVDed from A just like in TRG, truncated
    to dimension chi.
    """
    do_reuse = False
    if (pars["reuse_initial"] and
            pieces["w"] is not None and
            pieces["u"] is not None and
            pieces["v"] is not None):
        w_shp = pieces["w"].shape
        w_chi = type(A_list[0]).flatten_dim(w_shp[0])
        if (w_chi == chi and A_list[0].compatible_indices(pieces["u"], 1, 2)
                         and A_list[0].compatible_indices(pieces["w"], 0, 2)):
            do_reuse = True
    if do_reuse:
        u = pieces["u"].copy()
        v = pieces["v"].copy()
        w = pieces["w"].copy()
    else:
        # Some performance could be saved here if the As would not be SVDed
        # again. This would be simple if A_part_list elements didn't have
        # the S_sqrt multiplied into them.  The performance of this part
        # should be very much subleading though.
        # u:
        dim1 = A_list[0].shape[1]
        try:
            qim1 = A_list[0].qhape[1]
        except TypeError:
            qim1 = None
        dim2 = A_list[1].shape[1]
        try:
            qim2 = A_list[1].qhape[1]
        except TypeError:
            qim2 = None
        eye1 = type(A_list[0]).eye(dim1, qim=qim1)
        eye2 = type(A_list[0]).eye(dim2, qim=qim2)
        u = scon((eye1, eye2), ([-1,-3], [-4,-2]))
        # w:
        w_dg = A_list[0].svd((0,1), (2,3), chis=chi)[0]
        w = w_dg.transpose((2,1,0)).conjugate()
        # v:
        if pars["horz_refl"]:
            v = w_hat(w, gauges)
        else:
            v_dg = A_list[1].svd((0,3), (1,2), chis=chi)[2]
            v = v_dg.transpose((1,0,2)).conjugate()
    return u, v, w


def optimize_isometries(u, v, w, A_list, A_part_list, pars, gauges, chi=None):
    # Compute environments M_v and M_w
    if chi is None:
        chi_v = type(u).flatten_dim(v.shape[1])
        chi_w = type(u).flatten_dim(w.shape[0])
    else:
        chi_v = chi
        chi_w = chi
    if pars["print_errors"]>2:
        print('Optimizing isometries.')
    for i in range(pars["opt_iters_tens"]):
        upb = upper_block(u, v, w, A_list, A_part_list, pars)
        if pars["use_parts"]:
            # TODO Could figure out a way to do this in O(chi^6). I
            # wasn't able to come up with one and probably this won't be
            # used much, so I left it be.
            warnings.warn("In TNR, use_parts is True but is not utilized in "
                          "optimize_disentanglers. Reverting back to the "
                          "O(chi^7) algorithm.")
        # This is O(chi^7).
        env_top = scon((v, u, A_list[0], A_list[1]),
                       ([1,-3,3], [-2,1,4,2], [-1,4,5,-4], [5,2,3,-5]))
        env_top_dg = env_top.conjugate().transpose((0,3,4,2,1))
        M_w = scon((env_top, upb_dg(upb), upb, env_top_dg),
                   ([-1,-2,7,3,4], [3,4,2,1], [1,2,5,6], [-3,5,6,7,-4]))
        # Get the optimal isometry by SVDing the environment.
        S, U = M_w.eig((0,1), (2,3), chis=chi_w, hermitian=True,
                       print_errors=pars["print_errors"]-2)
        w = U.conjugate().transpose((2,1,0))
        if pars["horz_refl"]:
            v = w_hat(w, gauges)
        else:
            if pars["use_parts"]:
                # TODO see above.
                warnings.warn("In TNR, use_parts is True but is not utilized "
                              "in optimize_disentanglers. Reverting back to "
                              "the O(chi^7) algorithm.")
            # This is O(chi^7).
            env_top = scon((w, u, A_list[0], A_list[1]),
                           ([-1,1,2], [1,-2,3,4], [2,3,5,-4], [5,4,-3,-5]))
            env_top_dg = env_top.conjugate().transpose((3,4,2,1,0))
            M_v = scon((env_top, upb_dg(upb),
                        upb, env_top_dg),
                       ([5,-1,-2,3,4], [3,4,2,1],
                        [1,2,6,7], [6,7,-4,-3,5]))
            S, U = M_v.eig((0,1), (2,3), chis=chi_v, hermitian=True,
                           print_errors=pars["print_errors"]-2)
            v = U.conjugate().transpose((0,2,1))
            v = v.flip_dir(1)
    return v, w


def optimize_disentangler(u, v, w, A_list, A_part_list, pars,
                          return_B_norm=False):
    # Compute environment M
    for i in range(pars["opt_iters_tens"]):
        upb = upper_block(u, v, w, A_list, A_part_list, pars)
        # Note that by construction, B = B^dg.
        B = build_B(u, v, w, A_list, A_part_list, pars, upb=upb)
        if pars["use_parts"]:
            # This is O(chi^8), O(chi^6) if A can be split with just chi.
            M = scon((w, v,
                      A_part_list[0][0], A_part_list[0][1],
                      A_part_list[1][0], A_part_list[1][1],
                      upb_dg(upb), B),
                     ([10,-3,3], [-4,8,4],
                      [3,-1,9], [9,5,6],
                      [-2,4,11], [5,11,7],
                      [6,7,2,1], [1,2,8,10]))
        else:
            # This is O(chi^7).
            M = scon((w, v,
                      A_list[0], A_list[1],
                      upb_dg(upb), B),
                     ([3,-3,4], [-4,9,6],
                      [4,-1,7,5], [7,-2,6,8],
                      [5,8,2,1], [1,2,9,3]))
        U, S, V = M.svd((0,1), (2,3))
        u = scon((U.conjugate(), V.conjugate()), ([-3,-4,1], [1,-1,-2]))
        if pars["horz_refl"]:
            # u should be symmetric under a horizontal reflection
            # already, but we symmetrize to kill numerical errors.
            u = (u + u.conjugate().transpose((1,0,3,2))) / 2
    if return_B_norm:
        B_norm_sq = scon((M, u), ([1,2,3,4], [3,4,1,2])).value()
        if np.imag(B_norm_sq)/np.abs(B_norm_sq) > 1e-13:
            warnings.warn("B_norm_sq is complex: " + str(B_norm_sq))
        else:
            B_norm_sq = np.real(B_norm_sq)
        B_norm = np.sqrt(B_norm_sq)
        return u, B_norm
    else:
        return u


# Functions for executing the second step of TNR: Building and splitting
# B, building z, and putting it all together to form the new tensor.


def build_B(u, v, w, A_list, A_part_list, pars, upb=None):
    if upb is None:
        upb = upper_block(u, v, w, A_list, A_part_list, pars)
    B = scon((upb, upb_dg(upb)), ([-1,-2,1,2], [1,2,-3,-4]))
    return B


def split_B(B, pars):
    if pars["print_errors"]>0:
        print('-Splitting B.')
    if pars["horz_refl"]:
        S, U = B.eig((0,3), (1,2), chis=pars["chis_trg"], hermitian=True,
                     eps=pars["opt_eps_chi"],
                     print_errors=pars["print_errors"])
        # G_hh is a diagonal matrix populated with signs of the
        # eigenvalues of B.
        G_hh = S.sign()
        G_hh = G_hh.diag()
        S_sqrt = S.abs().sqrt()
        U = U.transpose((0,2,1))
        V = scon((U.conjugate(), G_hh), ([-2,1,-3], [-1,1]))
    else:
        U, S, V = B.svd((0,3), (1,2), chis=pars["chis_trg"],
                        eps=pars["opt_eps_chi"],
                        print_errors=pars["print_errors"])
        S_sqrt = S.sqrt()
        U = U.transpose((0,2,1))
        G_hh = None
    US = U.multiply_diag(S_sqrt, 1, direction="r")
    SV = V.multiply_diag(S_sqrt, 0, direction="l")
    return_value = (US, SV)
    if pars["return_gauges"]:
        return_value += (G_hh,)
    if pars["print_errors"]>0:
        print('Truncated bond dimension in split_B: %i' % len(S))
    return return_value


def build_z(v, w, BUS, BSV, pars):
    # This is O(chi^6).
    if pars["print_errors"]>0:
        print('-Building z.')
    # This could be tweaked a bit by using intermediate results
    BSV_dg = BSV.transpose((0,2,1)).conjugate()
    BUS_dg = BUS.transpose((2,1,0)).conjugate()
    M = scon((v_dg(v), w_dg(w),
              BSV, BUS,
              v_prime(v), w_prime(w), v_dg(v), w_dg(w),
              BSV_dg, BUS_dg,
              v_prime(v), w_prime(w)),
             ([-1,5,13], [5,-2,14],
              [7,13,11], [14,8,9],
              [11,1,3], [1,9,4], [3,2,12], [2,4,10],
              [7,12,15], [10,8,16],
              [15,6,-3], [6,16,-4]))
    M_diff = M - M.conjugate().transpose((2,3,0,1))
    S, U = M.eig((0,1), (2,3), chis=pars["chis_trg"], hermitian=True,
                 eps=pars["opt_eps_chi"], print_errors=pars["print_errors"])
    z = U.conjugate().transpose((2,0,1))
    if pars["print_errors"]>0:
        print('Truncated bond dimension in build_z: %i' % len(S))
    return z 


def build_A_new(v, w, z, BUS, BSV):
    # This is O(chi^6).
    A_new = scon((z, v_dg(v), w_dg(w), BSV, BUS,
                  v_prime(v), w_prime(w), z_dg(z)),
                 ([-2,3,4], [3,1,7], [1,4,10], [-1,7,9], [10,-3,8],
                  [9,2,5], [2,8,6], [5,6,-4]))
    return A_new


# Functions for fixing the gauge of the new tensor to be the same as the
# previous one.

def fix_A_new_gauge(A_new, A, pars, BUS, BSV, z, G_hh=None):
    if pars["print_errors"] > 1:
        print("Fixing A_new gauge.")
    dim_X = A_new.shape[1]
    try:
        qim_X = A_new.qhape[1]
    except TypeError:
        qim_X = None
    dim_Y = A_new.shape[0]
    try:
        qim_Y = A_new.qhape[0]
    except TypeError:
        qim_Y = None
    X = type(A_new).eye(dim_X, qim=qim_X, dtype=A_new.dtype)
    Y = type(A_new).eye(dim_Y, qim=qim_Y, dtype=A_new.dtype)
    A_norm = A.norm()
    orig_err = (A_new - A).norm() / A_norm
    cost = np.inf
    cost_change = np.inf
    counter = 0
    while cost_change > 1e-11 and counter < 10000:
        old_cost = cost
        X = fix_A_new_gauge_optimize_X(A_new, A, X, Y)
        Y, cost = fix_A_new_gauge_optimize_Y(A_new, A, X, Y, return_cost=True)
        cost_change = np.abs((old_cost - cost)/cost)
        counter += 1
    result = fix_A_new_gauge_update_to_gauge(A_new, BUS, BSV, z, X, Y,
                                             G_hh=G_hh)
    A_new = result[0]
    new_err = (A_new - A).norm() / A_norm
    if pars["print_errors"] > 1:
        print("After %i iterations, error in fix_A_new_gauge is %.3e. "
              "Original error was %.3e."%(counter, new_err, orig_err))
    return result

def fix_A_new_gauge_update_to_gauge(A_new, BUS, BSV, z, X, Y, G_hh=None):
    X_dg = matrix_dagger(X)
    Y_dg = matrix_dagger(Y)
    A_new = scon((A_new, Y, X, Y_dg, X_dg),
                 ([1,2,3,4], [-1,1], [-2,2], [3,-3], [4,-4]))
    BUS = scon((BUS, Y_dg), ([-1,2,-3], [2,-2]))
    BSV = scon((Y, BSV), ([-1,1], [1,-2,-3]))
    z = scon((X, z), ([-1,1], [1,-2,-3]))
    return_value = (A_new, BUS, BSV, z)
    if G_hh is not None:
        G_hh = scon((Y, G_hh, Y_dg), ([-1,1], [1,2], [2,-2]))
        return_value += (G_hh,)
    return return_value

def fix_A_new_gauge_optimize_X(A_new, A, X, Y):
    X_dg = matrix_dagger(X)
    Y_dg = matrix_dagger(Y)
    env = scon((A.conjugate(), Y, A_new, Y_dg, X_dg),
               ([6,-2,4,5], [6,2], [2,-1,1,3], [1,4], [3,5]))
    env_U, env_S, env_V = env.svd((0,), (1,))
    env_U_dg = matrix_dagger(env_U)
    env_V_dg = matrix_dagger(env_V)
    X = scon((env_V_dg, env_U_dg), ([-1,1], [1,-2]))
    return X

def fix_A_new_gauge_optimize_Y(A_new, A, X, Y, return_cost=False):
    X_dg = matrix_dagger(X)
    Y_dg = matrix_dagger(Y)
    env = scon((A.conjugate(), X, A_new, Y_dg, X_dg),
               ([-2,6,4,5], [6,2], [-1,2,1,3], [1,4], [3,5]))
    env_U, env_S, env_V = env.svd((0,), (1,))
    env_U_dg = matrix_dagger(env_U)
    env_V_dg = matrix_dagger(env_V)
    Y = scon((env_V_dg, env_U_dg), ([-1,1], [1,-2]))
    if return_cost:
        cost = scon((env, Y), ([1,2], [2,1])).value()
        if np.imag(cost) > 1e-13:
            warnings.warn("fix_A_new_gauge cost is complex: " + str(cost))
        else:
            cost = np.real(cost)
        return Y, cost
    else:
        return Y


# Functions related for extracting the gauge factors (Gs) related to
# different reflection symmetries of the final tensor.


def optimize_G_vh(A_new, pars):
    if pars["print_errors"] > 1:
        print("Optimizing G_vh.")
    Av = A_new.conjugate().transpose((0,3,2,1))
    dim = A_new.shape[0]
    try:
        qim = A_new.qhape[0]
    except TypeError:
        qim = None
    G_vh = type(A_new).eye(dim, qim=qim, dtype=A_new.dtype)
    G_vh = G_vh.flip_dir(0)
    cost = np.inf
    cost_change = np.inf
    counter = 0
    while cost_change > 1e-11 and counter < 10000:
        # The optimization step
        G_vh_dg = matrix_dagger(G_vh)
        env = scon((Av.conjugate(), A_new, G_vh_dg),
                   ([-2,2,4,3], [-1,2,1,3], [1,4]))
        env_U, env_S, env_V = env.svd((0,), (1,))
        env_U_dg = matrix_dagger(env_U)
        env_V_dg = matrix_dagger(env_V)
        G_vh = scon((env_V_dg, env_U_dg), ([-1,1], [1,-2]))

        old_cost = cost
        cost = scon((env, G_vh), ([1,2], [2,1])).value()
        if np.imag(cost) > 1e-13:
            warnings.warn("optimize_G_vh cost is complex: " + str(cost))
        else:
            cost = np.real(cost)
        cost_change = np.abs((old_cost - cost)/cost)
        counter += 1
    if pars["print_errors"] > 1:
        GAG_dg  = scon((G_vh, A_new, G_vh.conjugate().transpose()),
                       ([-1,1], [1,-2,3,-4], [3,-3]))
        err = (GAG_dg - Av).norm()
        print("After %i iterations, error in optimize_G_vh is %.3e."
              %(counter, err))
    return G_vh


def optimize_G_hv(A_new, G_hh, pars):
    if pars["print_errors"] > 1:
        print("Optimizing G_hv.")
    Ah = A_new.conjugate().transpose((2,1,0,3))
    dim = A_new.shape[1]
    try:
        qim = A_new.qhape[1]
    except TypeError:
        qim = None
    G_hv = type(A_new).eye(dim, qim=qim, dtype=A_new.dtype)
    G_hv = G_hv.flip_dir(0)
    cost = np.inf
    cost_change = np.inf
    counter = 0
    pre_env_A_new = scon((A_new, G_hh, G_hh), ([1,-2,3,-4], [-1,1], [3,-3]))
    while cost_change > 1e-11 and counter < 10000:
        # The optimization step
        G_hv_dg = matrix_dagger(G_hv)
        env = scon((pre_env_A_new, G_hv_dg, Ah.conjugate()),
                   ([3,-1,4,1], [1,2], [3,-2,4,2]))
        env_U, env_S, env_V = env.svd((0,), (1,))
        env_U_dg = matrix_dagger(env_U)
        env_V_dg = matrix_dagger(env_V)
        G_hv = scon((env_V_dg, env_U_dg), ([-1,1], [1,-2]))

        old_cost = cost
        cost = scon((env, G_hv), ([1,2], [2,1])).value()
        if np.imag(cost) > 1e-13:
            warnings.warn("optimize_G_hv cost is complex: " + str(cost))
        else:
            cost = np.real(cost)
        cost_change = np.abs((old_cost - cost)/cost)
        counter += 1
    if pars["print_errors"] > 1:
        GAG_dg  = scon((G_hv, pre_env_A_new, G_hv.conjugate().transpose()),
                       ([-2,2], [-1,2,-3,4], [4,-4]))
        err = (GAG_dg - Ah).norm()
        print("After %i iterations, error in optimize_G_hv is %.3e."
              %(counter, err))
    return G_hv


# Helper functions used in many places for building intermediate blocks
# of tensors, doing the various conjugate transposes of the isometries,
# etc.
#
# For the isometries the different subscripts are:
# - dg (dagger) for the diagonal conjugate transpose for which w w_dg =
#   v v_dg = 1.
# - taudg for the vertical conjugate transposes found in the lower corners
#   of the B tensor.
# - prime for the dg of a taudg. 
# - hat for the horizontal conjugate transposes needed when enforcing
#   left-right reflection symmetry.


def upper_block(u, v, w, A_list, A_part_list, pars):
    """ upper_block is the upper half of B. Often also called upb in the code.
    """
    if pars["use_parts"]:
        # This is O(chi^8), O(chi^6) if A can be split with just chi.
        upb = scon((u, v, w, A_part_list[0][0], A_part_list[0][1],
                      A_part_list[1][0], A_part_list[1][1]),
                     ([3,5,4,6], [5,-2,2], [-1,3,1], [1,4,7], [7,9,-3],
                      [6,2,8], [9,8,-4]))
    else:
        # This is O(chi^7).
        upb = scon((w, v, u, A_list[0], A_list[1]),
                   ([-1,6,2], [4,-2,1], [6,4,5,3], [2,5,7,-3], [7,3,1,-4]))
    return upb


def matrix_dagger(M):
    return M.transpose((1,0)).conjugate()


def u_dg(u):
    return u.transpose((2,3,0,1)).conjugate()


def v_dg(v):
    return v.transpose((0,2,1)).conjugate()


def v_taudg(v):
    return v.transpose((0,2,1)).conjugate()


def v_prime(v):
    return v.transpose((1,2,0))


def v_hat(v, gauges):
    v_hat = v.transpose((1,0,2)).conjugate()
    if gauges["G_hh"] is not None:
        v_hat = scon((v_hat, gauges["G_hh"]), ([-1,-2,3], [3,-3]))
    return v_hat


def w_dg(w):
    return w.transpose((2,1,0)).conjugate()


def w_taudg(w):
    return w.transpose((2,1,0)).conjugate()


def w_prime(w):
    return w.transpose((2,0,1))


def w_hat(w, gauges):
    w_hat = w.transpose((1,0,2)).conjugate()
    if gauges["G_hh"] is not None:
        w_hat = scon((w_hat, gauges["G_hh"]), ([-1,-2,3], [-3,3]))
    return w_hat


def z_dg(z):
    return z.transpose((1,2,0)).conjugate()


def upb_dg(upb):
    return upb.transpose((2,3,1,0)).conjugate()


# Functions for computing errors.


def A4_trace(A_list):
    # This is O(chi^6)
    A2 = scon(tuple(A_list), ([1,-1,2,-3], [2,-2,1,-4]))
    A4_trace_tensor = scon((A2, A2.conjugate()), ([1,2,3,4], [1,2,3,4]))
    A4_trace = A4_trace_tensor.norm()
    return A4_trace


def A4_frob_norm_sq(A_list, A_part_list, pars):
    # This is O(chi^6). Thus no need to utilize A_part_list even if
    # pars["use_parts"] is True.
    NW_corner = scon((A_list[0], A_list[0].conjugate()),
                     ([1,2,-1,-2], [1,2,-3,-4]))
    NE_corner = scon((A_list[1], A_list[1].conjugate()),
                     ([-1,1,2,-2], [-3,1,2,-4]))
    N_row = scon((NW_corner, NE_corner), ([1,-1,2,-3], [1,-2,2,-4]))
    norm_sq = N_row.norm_sq()
    return norm_sq


def compute_TNR_error(u, v, w, A_list, A_part_list, pars,
                      upb=None, B_norm=None, orig_norm=None):
    if orig_norm is None:
        orig_norm_sq = A4_frob_norm_sq(A_list, A_part_list, pars)
        orig_norm = np.sqrt(orig_norm_sq)
    else:
        orig_norm_sq = orig_norm**2
    if B_norm is None:
        if upb is None:
            upb = upper_block(u, v, w, A_list, A_part_list, pars)
        B = scon((upb, upb_dg(upb)), ([-1,-2,1,2], [1,2,-3,-4]))
        B_norm_sq = B.norm_sq()
    else:
        B_norm_sq = B_norm**2
    # The norm |TNR block - original block| can be reduced to this form
    diff_norm = np.sqrt(orig_norm_sq - B_norm_sq)
    err = diff_norm/orig_norm
    return err


def print_Z_error(A_list, log_fact, A_new, new_log_fact):
    """ Error in the partition function that is the trace of a block of
    4 A tensors.
    """
    A4_Z = A4_trace(A_list)
    A_new_Z_tensor = scon(A_new, [1,2,1,2]) * np.exp(new_log_fact-4*log_fact)
    A_new_Z = A_new_Z_tensor.norm()
    err = (A4_Z - A_new_Z)/A_new_Z 
    print('Relative difference in Z from exact A4 and from A_new:  %.3e + %gj'
          % (np.real(err), np.imag(err)))
    return err


# Functions for scaling a vertical string of tensors


def tnr_step_vertstr(A_list, log_fact_list=None, T_log_fact=0,
                     pars=dict(), gauges=dict(), pieces=dict(),
                     **kwargs):
    # Format the parameters and initialize A_part_list with the SVDs of
    # the original As.
    if log_fact_list is None:
        log_fact_list = [0]*len(A_list)
    A_list, pars, gauges, pieces = format_parameters_vertstr(\
            A_list, default_pars, default_gauges, pars, gauges, pieces, **kwargs)
    if pars["print_errors"] > 0:
        print("Beginning a TNR step for a vertical string.")
    A_part_list = split_A_list(A_list, pars)

    # Determine whether to use the SVD of the original tensor, based on
    # a rought estimate of whether it's computationally advantageous.
    A_NW = A_part_list[0][0]
    chi_orig = type(A_NW).flatten_dim(A_NW.shape[0])
    chi_split = type(A_NW).flatten_dim(A_NW.shape[2])
    pars["use_parts"] = chi_split < chi_orig**(3/2)
    if pars["print_errors"] > 1:
        if pars["use_parts"]:
            print("Using the SVDed A.")
        else:
            print("Not using the SVDed A.")

    # Obtain the optimized u, v, w.
    u, v, w = build_uvw(A_list, A_part_list, pars, gauges, pieces)

    # Build the intermediate tensors B and z and use them to put
    # together A_new.
    B = build_B(u, v, w, A_list, A_part_list, pars)
    split_B_result = split_B(B, pars)
    BUS, BSV = split_B_result[0:2]
    z1 = build_z(pieces["v"], w, BUS, pieces["BSV"], pars)
    z2 = build_z(v, pieces["w"], pieces["BUS"], BSV, pars)
    A_new1 = build_A_new(pieces["v"], w, z1, BUS, pieces["BSV"])
    A_new2 = build_A_new(v, pieces["w"], z2, pieces["BUS"], BSV)

    # Scale A_new to have largest values around unity and use that to
    # update new_log_fact.
    fact1 = A_new1.abs().max()
    if fact1 != 0:
        A_new1 = A_new1/fact1
        new_log_fact1 = np.log(fact1) + 2*T_log_fact + 2*log_fact_list[0]
    else:
        new_log_fact1 = 0
    fact2 = A_new2.abs().max()
    if fact2 != 0:
        A_new2 = A_new2/fact2
        new_log_fact2 = np.log(fact2) + 2*T_log_fact + 2*log_fact_list[1]
    else:
        new_log_fact2 = 0

    pieces = {'A_part_list': A_part_list, 'u': u, 'v': v, 'w': w,
              'B': B, 'BUS': BUS, 'BSV': BSV, 'z1': z1, 'z2': z2} 

    # Print error in trace of the block, if asked for.
    if pars["print_errors"] > 2:
        err = print_Z_error(A_list, log_fact, A_new, new_log_fact)

    # Put together the values to be returned and return.
    return (A_new1, A_new2), (new_log_fact1, new_log_fact2), pieces


def format_parameters_vertstr(A_list, default_pars, default_gauges,
                              pars, gauges, pieces, **kwargs):
    """ Formats some of the parameters given to tnr_step to a canonical
    form.
    """
    # Create pars.
    # Values are taken primarily from kwargs, then from pars and finally
    # from default. Only ones listed in defaults are used, others are
    # ignored.
    new_pars = default_pars.copy()
    new_pars.update(pars)
    for k in default_pars:
        if k in kwargs:
            new_pars[k] = kwargs[k]
    new_pars["horz_refl"] = False
    new_pars["A_eps"] = 0
    new_pars["fix_gauges"] = False
    new_pars["reuse_initial"] = False
    new_pars["return_pieces"] = False
    new_pars["return_gauges"] = False
    pars = new_pars

    new_gauges = default_gauges.copy()
    new_gauges.update(gauges)
    for k in default_gauges:
        if k in kwargs:
            new_gauges[k] = kwargs[k]
    gauges = new_gauges


    # Make sure chis_tnr and chis_trg are a lists of integers (or at
    # least singlet lists of one integer) and sorted from small to
    # large.
    if type(pars["chis_tnr"]) == int:
        pars["chis_tnr"] = [pars["chis_tnr"]]
    else:
        pars["chis_tnr"] = list(pars["chis_tnr"])
    pars["chis_tnr"] = sorted(pars["chis_tnr"])
    if type(pars["chis_trg"]) == int:
        pars["chis_trg"] = [pars["chis_trg"]]
    else:
        pars["chis_trg"] = list(pars["chis_trg"])
    pars["chis_trg"] = sorted(pars["chis_trg"])

    # If several chis to loop over are given but there is no epsilon to
    # determine sufficient accuracy, then just use the largest chi.
    if pars["opt_eps_chi"] == 0:
        pars["chis_tnr"] = [max(pars["chis_tnr"])]
        pars["chis_trg"] = [max(pars["chis_trg"])]
        pars["opt_eps_chi"] = np.inf

    A_list = list(A_list)
    if A_list[1].dirs == A_list[0].dirs:
        A_list[1] = A_list[1].flip_dir(1)
        A_list[1] = A_list[1].flip_dir(3)

    return A_list, pars, gauges, pieces

