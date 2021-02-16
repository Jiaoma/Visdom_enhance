import visdom
from typing import Optional, List, Any, Union, Mapping, overload, Text
from visdom import Visdom

import numpy as np

import torch

from os.path import join

import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve

from device_info import get_nvidia_smi,get_disk_usage


### Type aliases for commonly-used types.
# For optional 'options' parameters.
# The options parameters can be strongly-typed with the proposed TypedDict type once that is incorporated into the standard.
# See  http://mypy.readthedocs.io/en/latest/more_types.html#typeddict.
_OptOps = Optional[Mapping[Text, Any]]
_OptStr = Optional[Text]  # For optional string parameters, like 'window' and 'env'.

# No widely-deployed stubs exist at the moment for torch or numpy. When they are available, the correct type of the tensor-like inputs
# to the plotting commands should be
# Tensor = Union[torch.Tensor, numpy.ndarray, List]
# For now, we fall back to 'Any'.
Tensor = Any

# The return type of 'Visdom._send', which is turn is also the return type of most of the the plotting commands.
# It technically can return a union of several different types, but in normal usage,
# it will return a single string. We only type it as such to prevent the need for users to unwrap the union.
# See https://github.com/python/mypy/issues/1693.
_SendReturn = Text
'''
The main difference between Visdom and Visdom_E is that Vidsom_E can leave out the annoying counter and will automatically
judge whether it's the first time to paint a line or Image. As for Image, it support more type like PIL.Image and it
could automatically transform the channel order like CHW to HWC.
In the future I will develope more characters like performance analyze heatmap and tsne. 
'''
#use_incoming_socket=False
def pixel_values_in_mask(true_vessels, pred_vessels, masks, split_by_img=False):
    if split_by_img:
        n = pred_vessels.shape[0]
        return (np.array([true_vessels[i, ...][masks[i, ...] == 1].flatten() for i in range(n)]),
                np.array([pred_vessels[i, ...][masks[i, ...] == 1].flatten() for i in range(n)]))
    else:
        return true_vessels[masks == 1].flatten(), pred_vessels[masks == 1].flatten()
    
class Visdom_E(Visdom):
    def __init__(
            self,env=None
    ):
        if env!=None:
            super(Visdom_E, self).__init__(env=env)
        else:
            super(Visdom_E, self).__init__()
        self.clockDict={}

    def set(self,name,env=None):
        assert self.clockDict.get(name) == None
        self.clockDict[name] = {'clock': np.array([0]), 'env': env, 'win': None,'opt':name}

    def resetClock(self,name):
        assert name in self.clockDict
        self.clockDict[name]['clock']=np.array([0])

    def lineE(
        self,
        Y: Tensor,
        name:str,
        noAdd=False,
        withName=None,
        legend=None
    ):
        if isinstance(withName,type(None)):
            withName=name
        if isinstance(legend,type(None)):
            d=dict(title=self.clockDict[name]['opt'])
        else:
            d=dict(title=self.clockDict[name]['opt'],legend=list(legend))
        if isinstance(self.clockDict[name]['win'],type(None)):
            self.clockDict[name]['win']=super().line(Y,X=self.clockDict[name]['clock'],
                                                         env=self.clockDict[name]['env'],name=withName,opts=d)
        else:
            super().line(Y,X=self.clockDict[name]['clock'],win=self.clockDict[name]['win'],
                                                         env=self.clockDict[name]['env'],update='append',name=withName,opts=d)

        if not noAdd:
            self.clockDict[name]['clock'] += 1

    def imageE(
            self,
            img: Tensor,
            name: str,
            noAdd=False,
            needRegular=False
    ):
        if needRegular:
            img=(img-torch.min(img))/(torch.max(img)-torch.min(img)+1e-12)
        if self.clockDict[name]['clock'] == 0:
            self.clockDict[name]['win'] = super().image(img,env=self.clockDict[name]['env'],opts=dict(title=self.clockDict[name]['opt']))

        else:
            super().image(img,win=self.clockDict[name]['win'],env=self.clockDict[name]['env'],opts=dict(title=self.clockDict[name]['opt']))

        if not noAdd:
            self.clockDict[name]['clock'] += 1

    def tsne(self,
             batchXchannels,
             batchlabels,
             win=None,
             env=None,
             name=None,
             colors=None
             ):
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        cls=np.unique(batchlabels)
        print('Unique classes',str(cls))
        new_bc=[]
        for i in range(len(cls)):
            if (batchlabels==i).nonzero().shape[0]<100:
                new_bc.append(batchXchannels[batchlabels==i])
            else:
                new_bc.append(batchXchannels[batchlabels==i][:100])
        batchXchannels=np.concatenate(new_bc)
        # plot_only = 1000
        
        low_dim_embs = tsne.fit_transform(batchXchannels[:, :])
        labels = batchlabels[:]
        plt.cla()
        if colors is None:
            colors = ['black', 'red', 'green', 'blue', 'yellow', 'cyan', 'purple']
        else:
            colors=colors/255
        X, Y = low_dim_embs[:, 0], low_dim_embs[:, 1]
        for x, y, s in zip(X, Y, labels):
            # if s==0:
            #     continue
            c = cm.rainbow(int(255 * s // 9))
            # plt.text(x, y, s, backgroundcolor=c, fontsize=9)

            if type(colors)==np.ndarray:
                plt.scatter(x, y, c=colors[s].reshape(1,-1), s=16, lw=0)
            else:
                plt.scatter(x, y, c=colors[s], s=16, lw=0)
        plt.xlim(X.min(), X.max())
        plt.ylim(Y.min(), Y.max())
        if name==None:
            plt.title('t-SNE')
        else:
            plt.title(name)
        return self.matplot(plt,win=win,env=env)

    def tsneE(self,
        batchXchannels,
        batchlabels,
        name:str,
        noAdd=False,
        colors=None,
    ):
        if self.clockDict[name]['clock'] == 0:
            self.clockDict[name]['win'] = self.tsne(batchXchannels,batchlabels,env=self.clockDict[name]['env'],name=name,colors=colors)
        else:
            self.tsne(batchXchannels,batchlabels,win=self.clockDict[name]['win'],env=self.clockDict[name]['env'],name=name,colors=colors)

        if not noAdd:
            self.clockDict[name]['clock'] += 1

    def gpuDisk(
            self,
            name: str,
            noAdd=False
    ):
        try:
            gpu = get_nvidia_smi()
            disk = get_disk_usage()
            display = (gpu + '\n' + disk)
            display = display.replace('\n', '<br>')
        except:
            display="Warning: You have called gpuDisk in other place at the same time."

        if self.clockDict[name]['clock'] == 0:
            self.clockDict[name]['win'] = super().text(display,env=self.clockDict[name]['env'])

        else:
            super().text(display,win=self.clockDict[name]['win'],env=self.clockDict[name]['env'])

        if not noAdd:
            self.clockDict[name]['clock'] += 1

    #add measure functions
    def PRCurve(self,
                precision:np.ndarray,
                recall:np.ndarray,
                name:str,
                visName:str,
                plotStyle: str,
                bestPoint=None,
                savePath=None,
                noAdd=False):
        plt.cla()
        plt.title("Precision Recall Curve")

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.xlim(0.5, 1.0)
        plt.ylim(0.5, 1.0)
        if not isinstance(bestPoint,type(None)):
            plt.plot(recall, precision, label=name)
            plt.legend(loc='lower left')
        plt.plot(bestPoint[1], bestPoint[0], plotStyle, label=name + ' best f1')
        plt.legend(loc='lower left')

        if self.clockDict[visName]['clock'] == 0:
            self.clockDict[visName]['win'] = self.matplot(plt,win=self.clockDict[visName]['win'],env=self.clockDict[visName]['env'])

        else:
            self.matplot(plt,win=self.clockDict[visName]['win'],env=self.clockDict[visName]['env'])

        if not noAdd:
            self.clockDict[visName]['clock'] += 1
        if not isinstance(savePath,type(None)):
            plt.savefig(join(savePath, 'PRCurve.png'))


    def ROCCurve(self,
                 predict:np.ndarray,
                 groundTruth:np.ndarray,
                 name:str,
                 visName:str,
                 mask=None,
                 savePath=None,
                 noAdd=False):
        if not isinstance(mask,type(None)):
            groundTruth,predict=pixel_values_in_mask(groundTruth,predict,mask)
        plt.cla()
        plt.title("ROC Curve")
        plt.xlabel("fpr")
        plt.ylabel("tpr")
        plt.xlim(0.5, 1.0)
        plt.ylim(0.5, 1.0)
        fpr, tpr, _ = roc_curve(groundTruth.flatten(), predict.flatten())
        plt.plot(fpr, tpr, label=name)
        plt.legend(loc='lower left')
        if self.clockDict[visName]['clock'] == 0:
            self.clockDict[visName]['win'] = self.matplot(plt, win=self.clockDict[visName]['win'],
                                                          env=self.clockDict[visName]['env'])

        else:
            self.matplot(plt, win=self.clockDict[visName]['win'], env=self.clockDict[visName]['env'])

        if not noAdd:
            self.clockDict[visName]['clock'] += 1
        if not isinstance(savePath,type(None)):
            plt.savefig(join(savePath, 'ROCCurve.png'))


    def progressBar(
            self,
            percent: int,
            name: str,
            visName: str,
            noAdd=False
    ):
        barHTML='''<!DOCTYPE html>
                    <html>
                    <body>
                    
                    <h1>{}</h1>
                    
                    <div style="width: 100%;background-color: #ddd;text-align:center;">
                      <div style="width: {}%;height: 30px;background-color: #4CAF50;text-align: center;line-height: 30px;color: white;">{}%</div>
                    </div>
                    
                    </body>
                    </html>'''.format(name,percent,percent)
        if self.clockDict[visName]['clock'] == 0:
            self.clockDict[visName]['win'] = super().text(barHTML,env=self.clockDict[visName]['env'])

        else:
            super().text(barHTML,win=self.clockDict[visName]['win'],env=self.clockDict[visName]['env'])

        if not noAdd:
            self.clockDict[visName]['clock'] += 1

if __name__=='__main__':
    import time
    vis=Visdom_E()
    vis.set('progress',env='main')
    for i in range(100):
        vis.progressBar(i,'test','progress')
        time.sleep(1)

