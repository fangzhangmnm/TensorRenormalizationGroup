import numpy as np
import modeldata
import os
import itertools
from matplotlib import pyplot
from fractions import Fraction

filename = os.path.basename(__file__)
g_scalefact = 5
g_widthfact = 1
g_heightfact = 1.75
g_fontsize = 6.5*g_scalefact
g_tickfontsize = 5.5*g_scalefact
# The width of a two-column revtex column.  
g_figsize = (3.4*g_widthfact*g_scalefact, 1.7*g_heightfact*g_scalefact)
g_labelpad = 4.2*g_scalefact  # Default None
g_deg_circle_increase = 1.5*g_scalefact

pyplot.rc('font', size=g_fontsize)
#pyplot.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#pyplot.rc('text', usetex=True)

g_x_style_dict = {"marker":'x', "mew":0.6*g_scalefact,
                       "ms":3.35*g_scalefact, "ls":'', "color":'black'}
g_dot_style_dict = {"marker":'.', "mew":0.6*g_scalefact,
                       "ms":3.35*g_scalefact, "ls":'', "color":'black'}
g_circle_style_dict = {"marker":'o', "mew":0.4*g_scalefact,
                       "ms":5.2*g_scalefact, "ls":'', "color":'none',
                       "markeredgecolor":"green"}
g_bgline_style_dict = {"ls":"--", "color":"black", "alpha":0.25}
g_subplot_adjust_dict = {"bottom":0.1, "left":0.10, "right":0.95, "top":0.9}

def plot_and_print_dict(scaldims_dict, c, pars, pather, id_pars=None,
                        x_label="", y_label="", momenta_dict=None):
    for alpha, scaldims in scaldims_dict.items():
        print("\nFor defect angle %g"%alpha)
        if momenta_dict is not None:
            momenta = momenta_dict[alpha]
        else:
            momenta = None
        plot_and_print(scaldims, c, pars, pather, id_pars=id_pars, alpha=alpha,
                       momenta=momenta)
    if "plot_dict" in pars and pars["plot_dict"]:
        plot_dict(scaldims_dict, pars, x_label=x_label, y_label=y_label,
                  id_pars=id_pars)
    return None


def plot_and_print(scaldims, c, pars, pather, id_pars=None, alpha=None,
                   momenta=None):
    scaldims, momenta, max_dim = truncate_data(scaldims, pars,
                                               momenta=momenta)
    if pars["symmetry_tensors"] or "sep_qnums" in pars and pars["sep_qnums"]:
        exact_scaldims = {}
        exact_momenta = {}
        exact_degs = {}
        for k in scaldims.sects.keys():
            exact_scaldims[k], exact_momenta[k], exact_degs[k] =\
                modeldata.get_primary_data(max_dim+0.1, pars,
                                           alpha=alpha, qnum=k[0])
    else:
        exact_scaldims, exact_momenta, exact_degs = modeldata.get_primary_data(
            max_dim+0.1, pars, alpha=alpha
        )
    print_scaldims_and_momenta(scaldims, c, pars, momenta=momenta,
                               exact_scaldims=exact_scaldims,
                               exact_momenta=exact_momenta,
                               exact_degs=exact_degs)
    if pars["plot_by_qnum"]:
        plot_by_qnum(scaldims, pars, exact_scaldims=exact_scaldims,
                     exact_degs=exact_degs, alpha=alpha, id_pars=id_pars)
    if pars["plot_by_momenta"]:
        plot_by_momenta(scaldims, pars, momenta=momenta, alpha=alpha,
                        exact_scaldims=exact_scaldims,
                        exact_momenta=exact_momenta,
                        exact_degs=exact_degs, id_pars=id_pars)


def save_plot(fig, prefix, id_pars, **kwargs):
    # Save plot to file.
    pather = PathFinder(filename, id_pars, **kwargs)
    path = pather.generate_path(prefix, extension='.pdf')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    # Save pars to file.
    pars_path = pather.generate_path(prefix, extension=".pars")
    pars_f = open(pars_path, 'w')
    for k,v in sorted(id_pars.items()):
        print("%s = %s"%(k, v), file=pars_f)
    pars_f.close()


def get_subplot_nums(scaldims):
    num_qnums = len(scaldims.sects)
    subplots_xn = min(num_qnums, 4)
    subplots_yn = int(np.modf(num_qnums/subplots_xn)[1])
    if subplots_xn*subplots_yn < num_qnums:
        subplots_yn += 1
    return subplots_xn, subplots_yn


def print_nd(scaldims, pars, momenta=None, max_dim=0, qnum=None):
    print()
    if qnum is not None:
        print("Qnum: %i"%qnum)
    print("Scaling dimensions:")
    print(scaldims)
    if momenta is not None:
        print('with momenta:')
        print(momenta)


def get_new_figure():
    return pyplot.figure(figsize=g_figsize, facecolor="white")


def print_scaldims_and_momenta(scaldims, c, pars, momenta=None,
                               exact_scaldims=(), exact_momenta=(),
                               exact_degs=()):
    c_exact = modeldata.get_central_charge(pars)
    print('c:       ' + str(c))
    print('Exact c: ' + str(c_exact))

    if pars["symmetry_tensors"] or "sep_qnums" in pars and pars["sep_qnums"]:
        keys = sorted(scaldims.sects.keys())
        for k in keys:
            v = scaldims[k]
            m = momenta[k] if momenta is not None else None
            print_nd(v, pars, momenta=m, qnum=k[0])
    else:
        max_dim = print_nd(scaldims, pars, momenta=momenta)
    print("Exact scaldims:\n", exact_scaldims)
    print("with momenta:\n", exact_momenta)
    print("and degeneracies:\n", exact_degs)


def dim_grouper(v, tol=0.10):
    # Group v to elements that are at most tol appart.
    groups = []
    i = 0
    while i < len(v):
        s1 = v[i]
        j = i+1
        while j < len(v) and abs(v[j] - s1) < tol:
            j += 1
        groups.append(v[i:j])
        i = j
    return groups


def plot_nd_groups(groups, axes, max_group_size=None,
                   marker_dict=g_x_style_dict, return_max_group_size=False,
                   return_g_xs=False, **kwargs):
    if kwargs:
        marker_dict = marker_dict.copy()
        marker_dict.update(kwargs)
    if max_group_size is None:
        max_group_size = max(map(len, groups))
    # Find the right x-position for every s in v and plot.
    mid = max_group_size/2
    if return_g_xs:
        g_xs_list = []
    for g in groups:
        l = len(g)
        g_xs = np.arange(l, dtype=np.float_)
        g_xs += mid - (l-1)/2
        axes.plot(g_xs, g, **marker_dict)
        if return_g_xs:
            g_xs_list += list(g_xs)
    retval = ()
    if return_max_group_size:
        retval += (max_group_size,)
    if return_g_xs:
        retval += (g_xs_list,)
    if retval:
        if len(retval) > 1:
            return retval
        else:
            return retval[0]
    else:
        return


def plot_by_qnum_nd(v, pars, axes, exact_scaldims=(), exact_degs=(),
                    subplot_num=None, subplots_xn=None, qnum=None):
    # Multiply the exact_scaldims by degeneracies.
    exact_v = [[dim]*deg for dim, deg in zip(exact_scaldims, exact_degs)]
    exact_v = sum(exact_v, [])

    if exact_v and pars["draw_exact_circles"]:
        exact_groups = dim_grouper(exact_v)
        max_group_size, g_xs = plot_nd_groups(exact_groups, axes,
                                              marker_dict=g_circle_style_dict,
                                              return_max_group_size=True,
                                              return_g_xs=True)
        for s, x in zip(v, g_xs):
            axes.plot(x, s, **g_x_style_dict)
    else:
        groups = dim_grouper(v)
        max_group_size = plot_nd_groups(groups, axes,
                                        return_max_group_size=True)

    set_axes_props_qnum(axes, pars, v, exact_scaldims, max_group_size,
                        subplot_num=subplot_num, subplots_xn=subplots_xn,
                        qnum=qnum)


def set_axes_props_qnum(axes, pars, scaldims, exact_scaldims, n,
                        subplot_num=None, subplots_xn=None, qnum=None):
    x_low = 0
    x_high = n
    axes.get_xaxis().set_visible(False)
    set_axes_props_common(axes, pars, scaldims, exact_scaldims, x_low, x_high,
                          subplot_num=subplot_num, subplots_xn=subplots_xn,
                          qnum=qnum)


def plot_by_qnum(scaldims, pars, exact_scaldims=(), exact_degs=(),
                 alpha=None, id_pars=None):
    fig = get_new_figure()
    # Plot the scaling dimensions by qnum.
    if pars["symmetry_tensors"] or "sep_qnums" in pars and pars["sep_qnums"]:
        subplots_xn, subplots_yn = get_subplot_nums(scaldims)
        keys = sorted(scaldims.sects.keys())
        for i, k in enumerate(keys):
            q = k[0]
            v = scaldims[k]
            axes = fig.add_subplot(subplots_yn, subplots_xn, i+1)
            plot_by_qnum_nd(v, pars, axes, exact_scaldims=exact_scaldims[k],
                            exact_degs=exact_degs[k], subplot_num=i,
                            subplots_xn=subplots_xn, qnum=q)
    else:
        axes = fig.add_subplot(111)
        plot_by_qnum_nd(scaldims, pars, axes, exact_scaldims=exact_scaldims,
                        exact_degs=exact_degs)

    set_fig_props(fig, pars, alpha, fix_ylim=True)

    if pars["save_plots"]:
        save_plot(fig, "scaling_dims_by_qnum", id_pars, defect_angle=alpha)
    if pars["show_plots"]:
        pyplot.show()


def get_qnum_title_str(modelname, q):
    if modelname == "ising":
        qnum_title_str = "Parity %i"%(-1 if q else 1)
    elif modelname == "potts3":
        qnum_title_str = "$\mathbb{Z}_3$ charge %i"%q
    elif modelname == "sixvertex":
        qnum_title_str = "Particle number %i"%(q/2)
    else:
        qnum_title_str = "Quantum number sector %i"%q
    return qnum_title_str


def set_axes_props_common(axes, pars, scaldims, exact_scaldims, 
                          x_low, x_high, subplot_num=None, subplots_xn=None,
                          qnum=None):
    modelname = pars["model"].strip().lower()
    # Set qnum title.
    if qnum is not None:
        qnum_title_str = get_qnum_title_str(modelname, qnum)
        axes.set_title(qnum_title_str, fontsize=g_fontsize)
    # Name vertical axis.
    if subplot_num == 0 or subplot_num is None:
        axes.set_ylabel("Scaling dimension", labelpad=g_labelpad)
    # Set y ticks and hide borders.
    axes.tick_params(axis="both", which="both", top="off",  labelbottom="on",
                     left="off", right="off", labelleft="on",
                     labelsize=g_tickfontsize)
    #axes.spines["top"].set_visible(False)  
    #axes.spines["right"].set_visible(False)  
    if subplot_num == 1 and subplots_xn == 2:
        # If there's only two subplots, set the right one to have ticks
        # on the right. 
        pyplot.tick_params(axis="y", which="both", 
                           labelleft="off", labelright="on")
        #axes.spines["left"].set_visible(False)  
        #axes.spines["right"].set_visible(True)  
    exact_scaldims = tuple(set(exact_scaldims))
    axes.set_yticks(exact_scaldims)
    y_ticklabels_strs = get_ticklabels_from_exact(exact_scaldims)
    axes.set_yticklabels(y_ticklabels_strs)  
    axes.get_yaxis().tick_left()

    # Set y limits.
    #if modelname == "sixvertex":
    #    axes.set_ylim(bottom=scaldims[0]-0.1,
    #                  top=scaldims[-1]+0.1)
    #else:
    #    y_high = max(axes.get_ylim()[1], max(scaldims)+0.1)
    #    axes.set_ylim(bottom=-0.1, top=y_high)
    # Draw lines for exact_scaldims and set x limits.
    if pars["draw_exact_lines"]:
        for s in exact_scaldims:
            axes.plot((x_low, x_high), (s,s), scaley=False,
                      **g_bgline_style_dict)
            #axes.text(x_low-1, s, "s")
    axes.set_xlim(left=x_low, right=x_high)


def get_ticklabels_from_exact(exact):
    ticklabels = []
    for s in exact:
        f, i = np.modf(s)
        f = Fraction(f).limit_denominator()
        if f.denominator == 1:
            i += f
            f = 0
        ticklabel = ""
        if i != 0:
            ticklabel += "%i"%i
        if f != 0 or ticklabel == "":
            if f > 0 and ticklabel != "":
                ticklabel += "+"
            ticklabel += "%s"%str(f)
        ticklabels.append(ticklabel)
    return ticklabels


def set_axes_props_momenta(axes, pars, scaldims, exact_scaldims, momenta,
                           exact_momenta, subplot_num=None,
                           subplots_xn=None, qnum=None):
    axes.get_xaxis().tick_bottom()  
    axes.set_xlabel("Conformal spin", labelpad=g_labelpad)
    x_low = min(momenta) - 0.5
    x_high = max(momenta) + 0.5
    exact_momenta = tuple(set(exact_momenta))
    if "KW" in pars and pars["KW"]:
        # TODO This is ad hoc crap.
        exact_momenta = sorted(exact_momenta)
    axes.set_xticks(exact_momenta)
    x_ticklabels_strs = get_ticklabels_from_exact(exact_momenta)
    try:
        tickangle = pars["xtick_rotation"][subplot_num]
    except IndexError:
        tickangle = pars["xtick_rotation"][0]
    except TypeError:
        tickangle = pars["xtick_rotation"]
    if "KW" in pars and pars["KW"]:
        # TODO This is ad hoc crap.
        x_ticklabels_strs = [x if i%2==0 else ""
                             for i,x in enumerate(x_ticklabels_strs)]
    axes.set_xticklabels(x_ticklabels_strs, rotation=tickangle)
    if pars["draw_exact_lines"]:
        for s in exact_momenta:
            axes.plot((s, s), (scaldims.min()-1000, scaldims.max()+1000),
                      scaley=False, scalex=False, **g_bgline_style_dict)
    set_axes_props_common(axes, pars, scaldims, exact_scaldims, x_low, x_high,
                          subplot_num=subplot_num, subplots_xn=subplots_xn,
                          qnum=qnum)


def set_fig_props(fig, pars, alpha, fix_ylim=False):
    if ("defect_angles" in pars and pars["defect_angles"] != [0] and
            pars["draw_defect_angle"]):
        suptitle_str = "Defect angle = %g"%alpha
        fig.suptitle(suptitle_str)
    if pars["model"].strip().lower() == 'sixvertex':
        fig.tight_layout(pad=3, h_pad=2, w_pad=0.9)
    else:
        fig.tight_layout()
    if fix_ylim:
        mini = np.inf
        maxi = -np.inf
        minis, maxis = zip(*[ax.get_ylim() for ax in fig.axes])
        mini = min(minis) - 0.1
        maxi = max(maxis) + 0.1
        for ax in fig.axes:
            ax.set_ylim(bottom=mini, top=maxi)


def draw_exact_circles(axes, exact_scaldims, exact_momenta, exact_degs):
    style_dict = g_circle_style_dict.copy()
    del(style_dict["ms"])
    for momentum, dim, deg in zip(exact_momenta, exact_scaldims, exact_degs):
        for i in range(deg):
            ms = g_circle_style_dict["ms"] + i*g_deg_circle_increase
            axes.plot((momentum,), (dim,), ms=ms, scaley=False,
                      scalex=False, **style_dict)


def plot_by_momenta(scaldims, pars, momenta=None, exact_scaldims=(),
                    exact_momenta=(), exact_degs=(), alpha=None, id_pars=None):
    modelname = pars["model"].strip().lower()
    fig = get_new_figure()
    fig.subplots_adjust(**g_subplot_adjust_dict)
    # Plot the scaling dimensions by qnum and momenta.
    if pars["symmetry_tensors"] or "sep_qnums" in pars and pars["sep_qnums"]:
        subplots_xn, subplots_yn = get_subplot_nums(scaldims)
        keys = sorted(scaldims.sects.keys())
        for i, k in enumerate(keys):
            v = scaldims[k]
            m = momenta[k]
            q = k[0]
            if i>0 and modelname != "sixvertex":
                axes = fig.add_subplot(subplots_yn, subplots_xn, i+1)
            else:
                axes = fig.add_subplot(subplots_yn, subplots_xn, i+1)
            if pars["draw_exact_circles"]:
                draw_exact_circles(axes, exact_scaldims[k],
                                   exact_momenta[k], exact_degs[k])
            axes.plot(m, v, **g_x_style_dict)
            set_axes_props_momenta(axes, pars, v, exact_scaldims[k], m,
                                   exact_momenta[k], subplot_num=i,
                                   subplots_xn=subplots_xn,
                                   qnum=k[0])
    else:
        axes = fig.add_subplot(111)
        if pars["draw_exact_circles"]:
            draw_exact_circles(axes, exact_scaldims, exact_momenta, exact_degs)
        axes.plot(momenta, scaldims,  **g_x_style_dict)
        set_axes_props_momenta(axes, pars, scaldims, exact_scaldims, momenta,
                               exact_momenta)

    set_fig_props(fig, pars, alpha, fix_ylim=True)

    if pars["save_plots"]:
        save_plot(fig, "scaling_dims_by_momenta", id_pars, defect_angle=alpha)
    if pars["show_plots"]:
        pyplot.show()


def truncate_data(scaldims, pars, momenta=None):
    if pars["symmetry_tensors"] or "sep_qnums" in pars and pars["sep_qnums"]:
        max_dim = 0
        # Figure which qnums to plot and print.
        all_qnums = set(scaldims.qhape[0])
        if pars["qnums_plot"]:
            qnums = set(all_qnums) & set(pars["qnums_plot"])
        else:
            qnums = all_qnums
        # Truncate and leave out unwanted qnums.
        keys = tuple(scaldims.sects.keys())
        for k in keys:
            if k[0] not in qnums:
                del(scaldims[k])
                if momenta is not None:
                    del(momenta[k])
            else:
                v = scaldims[k]
                v = v[:pars["n_dims_plot"]]
                v = v[v < pars["max_dim_plot"]]
                scaldims[k] = v
                if momenta is not None:
                    momenta[k] = momenta[k][:len(v)]
                max_dim = max(max_dim, max(v))
    else:
        scaldims = scaldims[:pars["n_dims_plot"]]
        scaldims = scaldims[scaldims < pars["max_dim_plot"]]
        if momenta is not None:
            momenta = momenta[:len(scaldims)]
        max_dim = max(scaldims)
    return scaldims, momenta, max_dim


def plot_dict(scaldims_dict, pars, exacts_dict={}, exact_degs_dict={},
              x_label="", y_label="", id_pars=None):
    separate_parity = (pars["symmetry_tensors"]
                       or "sep_qnums" in pars and pars["sep_qnums"])
    even_color = "red" if separate_parity else "black"
    odd_color = "blue" if separate_parity else "black"

    fig = get_new_figure()
    axes = fig.add_subplot(111)
    alphas = sorted(scaldims_dict.keys())
    for alpha, scaldims in scaldims_dict.items():
        if hasattr(scaldims, "sects"):
            for qnum, sect in scaldims.sects.items():
                color = even_color if qnum == (0,) else odd_color
                style_dict = g_dot_style_dict.copy()
                style_dict.update(color=color)
                axes.plot([alpha]*len(sect), sect, **style_dict)
        else:
            axes.plot([alpha]*len(scaldims), scaldims, **g_dot_style_dict)

    if pars["draw_exact_circles"]:
        for alpha in exacts_dict:
            es = exacts_dict[alpha]
            if alpha in exact_degs_dict:
                degs = exact_degs_dict[alpha]
            else:
                degs = [1]*len(es)
            draw_exact_circles(axes, es, [alpha]*len(es), degs)

    # The following piece that computes the exact lines to draw includes
    # some ugly ad hoc fixes. Apologies.
    def alphatilde(theta):
        alphatilde = -theta/np.pi
        return alphatilde

    def alpha(theta):
        alpha = 0.5 + alphatilde(theta)
        return alpha

    thetas = scaldims_dict.keys()
    mintheta = min(thetas)
    maxtheta = max(thetas)
    maxangle = max(abs(alpha(maxtheta)), abs(alpha(mintheta)),
                   abs(alphatilde(maxtheta)), abs(alphatilde(mintheta)))
    
    maxdim = -np.inf
    for scaldims in scaldims_dict.values():
        scaldims.defval = -np.inf  # Ad hoc
        maxdim = max(maxdim, scaldims.max())

    maxint = 5
    maxk = 4
    tower1 = [(n,1) for n in range(maxint)]
    tower2 = [(n,-1) for n in range(1, maxint)]
    towers = tower1 + tower2
    mks = [(0,0)]
    for ts in itertools.product(towers, repeat=2):
        ts = set(ts)
        if len(ts) < 2:
            # There are duplicates, and fermions annihilate
            continue
        m = sum(t[0] for t in ts)
        k = sum(t[1] for t in ts)
        if m - np.abs(k)*maxangle < maxdim:
            mks.append((m,k))
    for ts in itertools.product(towers, repeat=4):
        ts = set(ts)
        if len(ts) < 4:
            # There are duplicates, and fermions annihilate
            continue
        m = sum(t[0] for t in ts)
        k = sum(t[1] for t in ts)
        if m - np.abs(k)*maxangle < maxdim:
            mks.append((m,k))
    for m, k in mks:
        # Even sector
        axes.plot([mintheta, maxtheta], [m + k*alpha(mintheta),
                                m + k*alpha(maxtheta)],
                  color=even_color,
                  scaley=False, scalex=False)
        # Odd sector
        axes.plot([mintheta, maxtheta], [m + k*alphatilde(mintheta)+1/8,
                                m + k*alphatilde(maxtheta)+1/4],
                  color=odd_color,
                  scaley=False, scalex=False)
    # End the ad hocness.

    axes.set_xlabel(x_label, labelpad=g_labelpad)
    axes.set_ylabel(y_label, labelpad=g_labelpad)
    
    ylim = axes.get_ylim()
    ylim = ylim[0] - 0.2, ylim[1]
    xlim = axes.get_xlim()
    xlim = xlim[0] - 0.02, xlim[1] + 0.02
    axes.set_ylim(ylim)
    axes.set_xlim(xlim)

    if pars["save_plots"]:
        save_plot(fig, "scaling_dims_dict", id_pars)
    if pars["show_plots"]:
        pyplot.show()

