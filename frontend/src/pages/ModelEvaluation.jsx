import {useEffect,useState} from 'react'
import {ChartNoAxesCombined,RefreshCw,FileText} from 'lucide-react'
import {BarChart,Bar,XAxis,YAxis,Tooltip,ResponsiveContainer,CartesianGrid,Legend} from 'recharts'
import api,{imageUrl} from '../services/api'
import LoadingSpinner from '../components/LoadingSpinner'

const modelLabel=name=>name==='original'?'Model A · Original':'Model B · Augmented'
const percent=value=>Number.isFinite(value)?`${(value*100).toFixed(1)}%`:'—'
const decimal=value=>Number.isFinite(value)?value.toFixed(3):'—'

export default function ModelEvaluation({notify}){
 const [data,setData]=useState(null),[busy,setBusy]=useState(false)
 const load=()=>api.get('/api/evaluation/results').then(r=>setData(r.data)).catch(e=>notify(e.message,'error'))
 useEffect(()=>{load()},[])
 const run=async()=>{
  setBusy(true)
  try{
   const r=await api.post('/api/evaluation/run');setData(r.data)
   if(r.data.errors?.length)notify(`Some evaluations failed: ${r.data.errors.map(x=>`${modelLabel(x.model)} — ${x.error}`).join('; ')}`,'error')
   else notify(r.data.pending?'Train both models before a full comparison.':'Both models evaluated on the shared test set.')
  }catch(e){notify(e.message,'error')}finally{setBusy(false)}
 }
 if(!data)return <LoadingSpinner label="Loading evaluation reports…"/>
 const models=data.models||[],metrics=data.comparison||[],errors=data.errors||[]
 return <div className="page-stack">
  <section className="page-heading"><div><div className="eyebrow">VALIDATION AND HELD-OUT TEST RESULTS</div><h1>Model evaluation</h1><p>Validation metrics are shown separately from final scores on the untouched test images.</p></div><button className="button primary" onClick={run} disabled={busy}><RefreshCw size={15} className={busy?'spin':''}/>{busy?'Evaluating…':'Run evaluation'}</button></section>
  {data.pending&&<div className="notice info">A full comparison needs both trained models. Available per-model reports are shown below.</div>}
  {errors.map(item=><div className="notice error" key={item.model}>{modelLabel(item.model)} evaluation failed: {item.error}</div>)}
  {metrics.length>0&&<section className="panel"><div className="panel-head"><div><h2>Metric summary</h2><p>Validation loss and accuracy; weighted precision, recall, and F1 are measured on the shared test set.</p></div></div><div className="evaluation-table-wrap"><table className="evaluation-table"><thead><tr><th>Model</th><th>Validation accuracy</th><th>Validation loss</th><th>Test accuracy</th><th>Weighted precision</th><th>Weighted recall</th><th>Weighted F1</th></tr></thead><tbody>{metrics.map(m=><tr key={m.model}><th>{modelLabel(m.model)}</th><td>{percent(m.validation_accuracy)}</td><td>{decimal(m.validation_loss)}</td><td>{percent(m.accuracy)}</td><td>{percent(m.precision)}</td><td>{percent(m.recall)}</td><td>{percent(m.f1_score)}</td></tr>)}</tbody></table></div></section>}
  <div className="content-grid"><section className="panel"><div className="panel-head"><div><h2><ChartNoAxesCombined size={18}/> Test metric comparison</h2><p>Accuracy and weighted precision, recall, and F1</p></div></div>{metrics.length?<ResponsiveContainer width="100%" height={290}><BarChart data={metrics}><CartesianGrid strokeDasharray="3 3" vertical={false}/><XAxis dataKey="model" tickFormatter={x=>x==='original'?'Model A':'Model B'}/><YAxis domain={[0,1]} tickFormatter={x=>`${Math.round(x*100)}%`}/><Tooltip formatter={x=>`${(x*100).toFixed(1)}%`}/><Legend/><Bar dataKey="accuracy" name="Accuracy" fill="#6867e8" radius={[4,4,0,0]}/><Bar dataKey="precision" name="Weighted precision" fill="#49aa97" radius={[4,4,0,0]}/><Bar dataKey="recall" name="Weighted recall" fill="#e6a64f" radius={[4,4,0,0]}/><Bar dataKey="f1_score" name="Weighted F1" fill="#e47f73" radius={[4,4,0,0]}/></BarChart></ResponsiveContainer>:<div className="empty-state">Run evaluation to calculate test metrics.</div>}</section>
   <section className="panel"><div className="panel-head"><div><h2><FileText size={18}/> Test-set details</h2><p>Per-model classification output</p></div></div>{models.length?models.map(m=><div className="report-row" key={m.model}><strong>{modelLabel(m.model)}</strong><span>{m.test_samples} test samples</span><details><summary>Classification report</summary><pre>{JSON.stringify(m.classification_report,null,2)}</pre></details><img className="confusion-image" src={imageUrl(`/files/outputs/confusion_matrix/${m.model}_confusion_matrix.png`)} alt={`${m.model} confusion matrix`}/></div>):<div className="empty-state">Classification reports will appear after evaluation.</div>}</section>
  </div>
 </div>
}
