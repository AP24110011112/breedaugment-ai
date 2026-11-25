import {useEffect,useState} from 'react'
import {CheckCircle2,AlertCircle,X} from 'lucide-react'
import Sidebar from './components/Sidebar'
import Navbar from './components/Navbar'
import Dashboard from './pages/Dashboard'
import Dataset from './pages/Dataset'
import Augmentation from './pages/Augmentation'
import ModelTraining from './pages/ModelTraining'
import ModelEvaluation from './pages/ModelEvaluation'
import Prediction from './pages/Prediction'
import About from './pages/About'
import api from './services/api'
const pages={dashboard:Dashboard,dataset:Dataset,augmentation:Augmentation,training:ModelTraining,evaluation:ModelEvaluation,prediction:Prediction,about:About}
export default function App(){
 const [page,setPage]=useState('dashboard'),[health,setHealth]=useState(false),[toast,setToast]=useState(null)
 const notify=(message,type='success')=>{setToast({message,type});window.setTimeout(()=>setToast(null),4500)}
 useEffect(()=>{api.get('/api/health').then(()=>setHealth(true)).catch(()=>setHealth(false));const id=setInterval(()=>api.get('/api/health').then(()=>setHealth(true)).catch(()=>setHealth(false)),10000);return()=>clearInterval(id)},[])
 const Page=pages[page]||Dashboard
 return <div className="app-shell"><Sidebar page={page} setPage={setPage}/><main className="main-area"><Navbar page={page} health={health}/><div className="main-content"><Page key={page} notify={notify}/></div><footer className="app-footer"><span>BreedAugment AI</span><span>Local-first · TensorFlow · FastAPI</span></footer></main>{toast&&<div className={`toast ${toast.type}`}><span>{toast.type==='error'?<AlertCircle size={18}/>:<CheckCircle2 size={18}/>}</span>{toast.message}<button aria-label="Dismiss notification" onClick={()=>setToast(null)}><X size={15}/></button></div>}</div>
}
