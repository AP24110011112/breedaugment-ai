import {useMemo,useState} from 'react'
import {Wand2,SlidersHorizontal,CheckCircle2,FolderPlus,Save,X,RefreshCw} from 'lucide-react'
import api from '../services/api'
import ImageUploader from '../components/ImageUploader'
import ImageComparison from '../components/ImageComparison'

const choices=[['lighting','Lighting','Brightness, contrast, tone, and subtle shadows'],['rotation','Rotation & flipping','Small pose changes and horizontal flips'],['perspective','Perspective','Mild viewpoint distortion'],['scaling','Scaling & shifts','Gentle zoom and translation'],['blur_noise','Blur & noise','Camera quality and sensor variation'],['occlusion','Occlusion','Small random cutout regions'],['combined','Combined','A balanced mixture of realistic transforms']]
const folderName=value=>value.trim().replace(/[^A-Za-z0-9_-]+/g,'_').replace(/^[_-]+|[_-]+$/g,'').slice(0,64)

export default function Augmentation({notify}){
 const [file,setFile]=useState(null),[strategy,setStrategy]=useState('combined'),[count,setCount]=useState(3)
 const [results,setResults]=useState([]),[selectedImages,setSelectedImages]=useState([]),[busy,setBusy]=useState(false)
 const [showSaveDialog,setShowSaveDialog]=useState(false),[saving,setSaving]=useState(false),[breeds,setBreeds]=useState([])
 const [categoryChoice,setCategoryChoice]=useState(''),[newBreed,setNewBreed]=useState(''),[createCategory,setCreateCategory]=useState(true),[savedSummary,setSavedSummary]=useState(null)
 const original=useMemo(()=>file?URL.createObjectURL(file):null,[file])
 const onFileChange=value=>{setFile(value);setResults([]);setSelectedImages([]);setSavedSummary(null)}
 const generate=async e=>{
  e.preventDefault()
  if(!file)return notify('Choose an image to augment.','error')
  const data=new FormData();data.append('image',file);data.append('strategy',strategy);data.append('count',String(count));setBusy(true)
  try{const r=await api.post('/api/augmentation/generate',data,{headers:{'Content-Type':'multipart/form-data'}});setResults(r.data.images);setSelectedImages(r.data.images.map(item=>item.filename));setSavedSummary(null);notify(`Generated ${r.data.count} augmentation(s).`)}
  catch(err){notify(err.message,'error')}finally{setBusy(false)}
 }
 const toggleImage=filename=>setSelectedImages(current=>current.includes(filename)?current.filter(item=>item!==filename):[...current,filename])
 const openSaveDialog=async()=>{
  try{
   const r=await api.get('/api/dataset/breeds');const items=r.data.breeds||[];setBreeds(items)
   const stem=(file?.name||'').replace(/\.[^.]+$/,'').toLowerCase()
   const inferred=items.find(item=>stem===item.breed.toLowerCase()||stem.startsWith(`${item.breed.toLowerCase()}_`))
   setCategoryChoice(inferred?.breed||'');setNewBreed('');setCreateCategory(true);setShowSaveDialog(true)
  }catch(error){notify(error.message,'error')}
 }
 const isNewCategory=categoryChoice==='__new__'
 const selectedBreed=isNewCategory?folderName(newBreed):categoryChoice
 const destination=selectedBreed?`dataset/original/${selectedBreed}/`:''
 const save=async()=>{
  if(!selectedBreed)return notify('Select an existing breed or enter a new category name.','error')
  if(isNewCategory&&!createCategory)return notify('Choose to create the new category folder before saving.','error')
  const selected=results.filter(item=>selectedImages.includes(item.filename))
  if(!selected.length)return notify('Select at least one augmented image.','error')
  setSaving(true)
  try{
   const r=await api.post('/api/augmentation/add-to-dataset',{breed:selectedBreed,original_filename:file?.name||'pet-image.jpg',images:selected.map(item=>({filename:item.filename,strategy:item.strategy||strategy})),create_category:isNewCategory&&createCategory})
   setSavedSummary(r.data);setShowSaveDialog(false);notify('Augmented images successfully added to the dataset.','success')
  }catch(error){notify(error.message,'error')}finally{setSaving(false)}
 }
 const allSelected=results.length>0&&selectedImages.length===results.length
 return <div className="page-stack">
  <section className="page-heading"><div><div className="eyebrow">IMAGE LAB</div><h1>Augmentation studio</h1><p>Preview controlled image transformations before adding variations to your training set.</p></div><div className="lab-badge"><Wand2 size={15}/> Albumentations</div></section>
  <div className="content-grid augmentation-layout">
   <section className="panel"><div className="panel-head"><div><h2>Configure transformation</h2><p>Use a clean original photo for a meaningful preview.</p></div></div><form onSubmit={generate} className="form-stack">
    <ImageUploader file={file} onChange={onFileChange} label="Upload an original pet image"/>
    <label className="field-label">Augmentation strategy<select className="text-input" value={strategy} onChange={e=>setStrategy(e.target.value)}>{choices.map(([id,label])=><option value={id} key={id}>{label}</option>)}</select></label>
    <div className="strategy-info"><SlidersHorizontal size={16}/><span>{choices.find(x=>x[0]===strategy)?.[2]}</span></div>
    <label className="field-label">Number of variations <div className="range-row"><input type="range" min="1" max="12" value={count} onChange={e=>setCount(Number(e.target.value))}/><strong>{count}</strong></div></label>
    <button className="button primary full" disabled={busy}>{busy?'Generating…':'Generate variations'}</button><p className="hint">Previews stay outside the dataset until you choose Add to Dataset.</p>
   </form></section>
   <section className="panel preview-panel"><div className="panel-head"><div><h2>Side-by-side preview</h2><p>Select the variations you want to keep.</p></div>{results.length>0&&<div className="preview-actions"><span className="success-chip"><CheckCircle2 size={14}/> {selectedImages.length} selected</span><button className="button small secondary" onClick={()=>setSelectedImages(allSelected?[]:results.map(item=>item.filename))}>{allSelected?'Deselect all':'Select all'}</button></div>}</div>
    {file&&results.length?<ImageComparison original={original} generated={results} selectable selectedIds={selectedImages} onToggle={toggleImage}/>:<div className="preview-empty"><div className="preview-symbol"><Wand2 size={28}/></div><strong>Your preview will appear here</strong><span>Upload a photo, select a strategy, and generate variations.</span></div>}
    {results.length>0&&<div className="add-dataset-actions"><button className="button primary" onClick={openSaveDialog} disabled={!selectedImages.length||busy}><FolderPlus size={15}/> Add to Dataset ({selectedImages.length})</button></div>}
    {savedSummary&&<div className="notice success save-result"><strong>Augmented images successfully added to the dataset.</strong><span>{savedSummary.total_images} total · {savedSummary.original_images} original · {savedSummary.augmented_images} augmented in {savedSummary.breed.replaceAll('_',' ')}.</span></div>}
   </section>
  </div>
  {showSaveDialog&&<div className="modal-backdrop" onMouseDown={e=>{if(e.target===e.currentTarget&&!saving)setShowSaveDialog(false)}}><section className="save-dialog" role="dialog" aria-modal="true" aria-labelledby="save-dialog-title">
   <div className="save-dialog-head"><div><div className="eyebrow">SAVE PREVIEWS</div><h2 id="save-dialog-title">Add images to dataset</h2></div><button className="icon-button" onClick={()=>setShowSaveDialog(false)} disabled={saving} aria-label="Close"><X size={18}/></button></div>
   <div className="form-stack"><label className="field-label">Breed or dataset category<select className="text-input" value={categoryChoice} onChange={e=>setCategoryChoice(e.target.value)}><option value="">Select a category…</option>{breeds.map(item=><option value={item.breed} key={item.breed}>{item.breed.replaceAll('_',' ')}</option>)}<option value="__new__">Create a new category…</option></select></label>
    {isNewCategory&&<><label className="field-label">New category name<input className="text-input" value={newBreed} onChange={e=>setNewBreed(e.target.value)} placeholder="e.g. golden_retriever" maxLength={64}/></label><label className="create-folder-option"><input type="checkbox" checked={createCategory} onChange={e=>setCreateCategory(e.target.checked)}/> Create the category folder if it does not exist</label></>}
    <div className="save-summary"><h3>Before saving</h3><dl><dt>Original image</dt><dd title={file?.name}>{file?.name}</dd><dt>Selected category</dt><dd>{selectedBreed?selectedBreed.replaceAll('_',' '):'Choose a category above'}</dd><dt>Augmented images</dt><dd>{selectedImages.length}</dd><dt>Destination folder</dt><dd>{destination||'Select a category to preview destination'}</dd></dl></div>
    {saving&&<div className="saving-progress" role="status"><RefreshCw size={16} className="spin"/><span>Saving images to the dataset…</span><div className="indeterminate-track"><div/></div></div>}
    <div className="button-row"><button className="button secondary" onClick={()=>setShowSaveDialog(false)} disabled={saving}>Cancel</button><button className="button primary" onClick={save} disabled={saving||!selectedBreed||!selectedImages.length||(isNewCategory&&!createCategory)}><Save size={15}/>{saving?'Saving…':'Confirm and save'}</button></div>
   </div>
  </section></div>}
 </div>
}
