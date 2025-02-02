# Function to compute Detection Cost Function (DCF).
def compute_dcf(y_true, y_pred, C_miss=1, C_fa=1, P_target=0.5):
    y_pred = (y_pred >= 0.5).float()
    fn = ((y_true == 1) & (y_pred == 0)).sum().item()
    fp = ((y_true == 0) & (y_pred == 1)).sum().item()
    tn = ((y_true == 0) & (y_pred == 0)).sum().item()
    tp = ((y_true == 1) & (y_pred == 1)).sum().item()

    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    dcf = C_miss * P_target * fnr + C_fa * (1 - P_target) * fpr
    return dcf
