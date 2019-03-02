from sklearn.metrics import ranking, confusion_matrix
import matplotlib.pyplot as plt
import itertools
import numpy as np

def show_results(y_test, prob_test, name, show=True, output_folder='', maxFNR=0.03, thresh = None):
    auc = ranking.roc_auc_score(y_test, prob_test, average=None, sample_weight=None)

    fpr, tpr, thresholds = ranking.roc_curve(y_test, prob_test, pos_label=1, sample_weight=None)
    fnr = 1 - tpr

    eer = min(zip(fpr, fnr, thresholds), key=lambda x: abs(x[0] - x[1]))

    idx_fnr = np.where(fnr<maxFNR)[0][0]
    if thresh == None:
        target_fnr = thresholds[idx_fnr]
    else:
        target_fnr = thresh
    y_pred = [float(score>=target_fnr) for score in prob_test]

    #fig = plt.figure()

    # show ROC
    if show:
        plt.figure(221)
        plt.plot(fpr, tpr, linewidth=2)
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title(name + ' - ROC curve, AUC = %f' % (auc))

        # show FPR-FNR vs threshold curves
        plt.figure(222)
        fnr_line, = plt.plot(thresholds, fnr * 100, linewidth=2, color='blue')
        fpr_line, = plt.plot(thresholds, fpr * 100, linewidth=2, color='red', linestyle='--')
        plt.legend([fnr_line, fpr_line], ['False Negative Rate (FNR)', 'False Positive Rate (FPR)'])
        plt.ylim(0, 100.001)
        plt.xlim(np.min(prob_test), np.max(prob_test))
        plt.title(name + ' - EER = %0.1f%% at t=%0.2f' % (100 * (eer[0] + eer[1]) / 2, eer[2]))
        plt.show()

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print ('AUC = %.2f' % (auc))
    print ('Confusion matrix (absolute frequency) at threshold = %.2f' % (target_fnr))
    print ('+---------------+------------+------------+')
    print ('|               |          TRUTH          |')
    print ('+---------------+------------+------------+')
    print ('|   PREDICTED   |  LEGIT(1)  |  FAKE (0)  |')
    print ('+---------------+------------+------------+')
    print ('|    LEGIT (1)  |%12d|%12d|' % (tp, fp))
    print ('+---------------+------------+------------+')
    print ('|     FAKE (0)  |%12d|%12d|' % (fn, tn))
    print ('+---------------+------------+------------+')

    print ('Confusion matrix (relative to |LEGIT| and |FAKE|) at threshold = %.2f' % (target_fnr))
    print ('+---------------+------------+------------+')
    print ('|               |          TRUTH          |')
    print ('+---------------+------------+------------+')
    print ('|   PREDICTED   |  LEGIT(1)  |  FAKE (0)  |')
    print ('+---------------+------------+------------+')
    print ('|    LEGIT (1)  |%11.1f%%|%11.1f%%|' % (tp*100.0/(tp+fn), fp*100.0/(fp+tn)))
    print ('+---------------+------------+------------+')
    print ('|     FAKE (0)  |%11.1f%%|%11.1f%%|' % (fn*100.0/(tp+fn), tn*100.0/(fp+tn)))
    print ('+---------------+------------+------------+')

    return y_pred, target_fnr