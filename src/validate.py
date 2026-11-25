"""Heuristic checks for generated images."""
import numpy as np
from config import REPORTS
from utils.io import read_image,write_json

def validate(generated):
    results=[]
    for row in generated:
        status='ok'; issue=None; difference=None
        try:
            image=read_image(row['path']); a=np.asarray(image,dtype=np.float32)
            if image.size[0]<32 or image.size[1]<32: status='too_small'
            source=read_image(row['source']).resize(image.size)
            difference=float(np.abs(a-np.asarray(source,dtype=np.float32)).mean())
            if difference<.25: status='nearly_unchanged'
            stats={'width':image.width,'height':image.height,'mean_pixel':float(a.mean()),'pixel_std':float(a.std())}
        except Exception as exc: status='unreadable'; issue=str(exc); stats={}
        results.append({'path':row['path'],'status':status,'mean_absolute_difference':difference,'stats':stats,'error':issue})
    report={'total':len(results),'valid':sum(r['status']=='ok' for r in results),'issues':sum(r['status']!='ok' for r in results),'results':results,'note':'Pixel checks are heuristic and cannot confirm the pet remains semantically intact or promise better accuracy.'}
    write_json(report,REPORTS/'augmentation_validation.json'); return report
