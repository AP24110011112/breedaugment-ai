import {LoaderCircle} from 'lucide-react'
export default function LoadingSpinner({label='Loading...'}){return <div className="loading"><LoaderCircle className="spin" size={20}/><span>{label}</span></div>}
