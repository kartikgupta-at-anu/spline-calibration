import numpy as np
import matplotlib.pyplot as plt
import os
import utilities as utils


def ensure_numpy(a):
    if not isinstance(a, np.ndarray): a = a.numpy()
    return a


def compute_accuracy (scores_in, labels_in, spline_method, splines, outdir, plotname, showplots=True) :

    # Computes the accuracy given scores and labels.
    # Also plots a graph of the spline fit

    # Change to numpy, then this will work
    scores = ensure_numpy (scores_in)
    labels = ensure_numpy (labels_in)

    # Sort them
    order = np.argsort(scores)
    scores = scores[order]
    labels = labels[order]

    #Accumulate and normalize by dividing by num samples
    nsamples = utils.len0(scores)
    integrated_accuracy = np.cumsum(labels) / nsamples
    integrated_scores   = np.cumsum(scores) / nsamples
    percentile = np.linspace (0.0, 1.0, nsamples)

    # Now, try to fit a spline to the accumulated accuracy
    nknots = splines
    kx = np.linspace (0.0, 1.0, nknots)

    error = integrated_accuracy - integrated_scores
    #error = integrated_accuracy

    spline = utils.Spline (percentile, error, kx, runout=spline_method)

    # Now, compute the accuracy at the original points
    dacc = spline.evaluate_deriv (percentile)
    #acc = dacc
    acc = scores + dacc

    # Compute the error
    fitted_error = spline.evaluate (percentile)
    err = error - fitted_error
    stdev = np.sqrt(np.mean(err*err))
    print (f"compute_error: fitted spline with accuracy {utils.str(stdev, form='{:.3e}')}")

    if showplots :
        # Set up the graphs
        f, ax = plt.subplots()
        f.suptitle ("Spline-fitting")

        # (accumualated) integrated_scores and # integrated_accuracy vs sample number
        ax.plot(100.0*percentile, error, label='Error')
        ax.plot(100.0*percentile, fitted_error, label='Fitted error')
        ax.legend()
        plt.savefig(os.path.join(outdir, plotname) + '_splinefit.png', bbox_inches="tight")
        plt.close()
    return acc, -fitted_error


def plot_KS_graphs(scores, labels, spline_method, splines, outdir, plotname, title="", showplots=True):
    # KS stands for Kolmogorov-Smirnov
    # Plots a graph of the scores and accuracy
    tks = utils.Timer ("Plotting graphs")

    # Change to numpy, then this will work
    scores = ensure_numpy (scores)
    labels = ensure_numpy (labels)

    # Sort the data
    order = scores.argsort()
    scores = scores[order]
    labels = labels[order]

    # Accumulate and normalize by dividing by num samples
    nsamples = utils.len0(scores)
    integrated_scores = np.cumsum(scores) / nsamples
    integrated_accuracy   = np.cumsum(labels) / nsamples
    percentile = np.linspace (0.0, 1.0, nsamples)
    fitted_accuracy, fitted_error = compute_accuracy (scores, labels, spline_method, splines, outdir, plotname, showplots=showplots)

    # Work out the Kolmogorov-Smirnov error
    KS_error_max = np.amax(np.absolute (integrated_scores - integrated_accuracy))

    if showplots:
        # Set up the graphs
        f, ax = plt.subplots(1, 4, figsize=(20, 5))
        size = 0.2
        f.suptitle (title+ f"\nKS-error = {utils.str(float(KS_error_max)*100.0)}%, "
                           f"Probability={utils.str(float(integrated_accuracy[-1])*100.0)}%"
                    , fontsize=18, fontweight="bold")

        # First graph, (accumualated) integrated_scores and integrated_accuracy vs sample number
        ax[0].plot(100.0*percentile, integrated_scores, linewidth=3, label='Cumulative Score')
        ax[0].plot(100.0*percentile, integrated_accuracy, linewidth=3, label='Cumulative Probability')
        ax[0].set_xlabel("Percentile", fontsize=16, fontweight="bold")
        ax[0].set_ylabel("Cumulative Score / Probability", fontsize=16, fontweight="bold")
        ax[0].legend(fontsize=13)
        ax[0].set_title('(a)', y=-size, fontweight="bold", fontsize=16) # increase or decrease y as needed
        ax[0].grid()

        # Second graph, (accumualated) integrated_scores and integrated_accuracy versus
        # integrated_scores
        ax[1].plot(integrated_scores, integrated_scores, linewidth=3, label='Cumulative Score')
        ax[1].plot(integrated_scores, integrated_accuracy, linewidth=3,
                   label="Cumulative Probability")
        ax[1].set_xlabel("Cumulative Score", fontsize=16, fontweight="bold")
        # ax[1].set_ylabel("Cumulative Score / Probability", fontsize=12)
        ax[1].legend(fontsize=13)
        ax[1].set_title('(b)', y=-size, fontweight="bold", fontsize=16) # increase or decrease y as needed
        ax[1].grid()

        # Third graph, scores and accuracy vs percentile
        ax[2].plot(100.0*percentile, scores, linewidth=3, label='Score')
        ax[2].plot(100.0*percentile, fitted_accuracy, linewidth=3, label=f"Probability")
        ax[2].set_xlabel("Percentile", fontsize=16, fontweight="bold")
        ax[2].set_ylabel("Score / Probability", fontsize=16, fontweight="bold")
        ax[2].legend(fontsize=13)
        ax[2].set_title('(c)', y=-size, fontweight="bold", fontsize=16) # increase or decrease y as needed
        ax[2].grid()

        # Fourth graph,
        # integrated_scores
        ax[3].plot(scores, scores, linewidth=3, label=f"Score")
        ax[3].plot(scores, fitted_accuracy, linewidth=3, label='Probability')
        ax[3].set_xlabel("Score", fontsize=16, fontweight="bold")
        # ax[3].set_ylabel("Score / Probability", fontsize=12)
        ax[3].legend(fontsize=13)
        ax[3].set_title('(d)', y=-size, fontweight="bold", fontsize=16) # increase or decrease y as needed
        ax[3].grid()
        plt.savefig(os.path.join(outdir, plotname) + '_KS.pdf', bbox_inches="tight")
        plt.close()
        tks.free()
    return KS_error_max
