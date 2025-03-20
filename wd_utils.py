

from otdd.pytorch.distance import  FeatureCost
from otdd.pytorch.moments import *
from otdd.pytorch.utils import *
import ot

def cal_distance(X,Y,metric):
    
    nx, ny  = X.shape[0], Y.shape[0]
    p = 2 if metric=='sqeuclidean' else 1  
    if a is None:  
        a = np.ones((nx,),dtype=np.float64) / nx
    if b is None:
        b = np.ones((ny,),dtype=np.float64) / ny  
    # loss matrix
    M = ot.dist(X,Y,metric=metric) # squared euclidean distance 'default'
    # compute EMD
    norm = np.max(M) if np.max(M)>1 else 1
    G0 = ot.emd(a, b, M/norm)
    
    cost = np.sum(G0*M)**(1/p)

    return cost 
    
class InterpMeas:
    def __init__(self, metric: str = "sqeuclidean", t_val: float = 0.5):
        self.metric = metric
        self.t_val = t_val

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        # a: np.ndarray | None = None,
        # b: np.ndarray | None = None,
    ):
        """

        Args:
            X `numpy.ndarray`
            Y `numpy.ndarray`
            a `numpy.ndarray` | `NONE` The weights of the empirical distribution X . Defaults to None with equal weights.
            b `numpy.ndarray` | `NONE` The weights of the empirical distribution X . Defaults to None with equal weights.

        Returns:
        """

        t_val = np.random.rand() if self.t_val == None else self.t_val

        nx, ny = X.shape[0], Y.shape[0]
        p = 2 if self.metric == "sqeuclidean" else 1

        a = np.ones((nx,), dtype=np.float64) / nx
        b = np.ones((ny,), dtype=np.float64) / ny

        M = ot.dist(X, Y, metric=self.metric)

        norm = np.max(M) if np.max(M) > 1 else 1
        G0 = ot.emd(a, b, M / norm)

        Z = (1 - t_val) * X + t_val * (G0 * nx) @ Y

        return Z
   


def compute_test(model, test_loader, moon_model=False, device="cpu"):
        
    if model.training:
        model.eval()

    correct, total = 0, 0
    criterion = nn.CrossEntropyLoss().to(device)
    loss_collector = []

    with torch.no_grad():
    
        for batch_idx, (x, target) in enumerate(test_loader):
            x, target = x.to(device), target.to(device,dtype=torch.int64)
            if moon_model:
                _, _, out = model(x)
            else:
                out = model(x)
            _, pred_label = torch.max(out.data, 1)
            
            loss = criterion(out, target)
            loss_collector.append(loss.item())

            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()

        test_acc = correct/float(total)
        test_loss = sum(loss_collector) / len(loss_collector)

    return test_acc, test_loss