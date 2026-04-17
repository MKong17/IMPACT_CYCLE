import time
from tools import sam_infer_core as s
img='tmp/sam3_probe.png'
cfg=s._load_json('configs/sam3_runtime.linux.json')
print('STEP 1 load_json', flush=True)
repo=s._runtime_repo_root(cfg)
print('STEP 2 repo', repo, flush=True)
api=s._load_sam3_api(repo)
print('STEP 3 load_api ok', flush=True)
st=s._sam3_stage_runtime_config(cfg,'category_discovery')
print('STEP 4 stage cfg ready', flush=True)
t0=time.time()
bundle=s._get_sam3_model_bundle(api,repo,st)
print('STEP 5 model bundle loaded in', time.time()-t0, flush=True)
proc=bundle['processor']
pil=api['PIL_Image']
im=pil.open(img).convert('RGB')
print('STEP 6 image loaded', im.size, flush=True)
state=proc.set_image(im)
print('STEP 7 set_image done', flush=True)
state=dict(state)
state['backbone_out']=dict(state.get('backbone_out')) if isinstance(state.get('backbone_out'),dict) else state.get('backbone_out')
state=proc.set_text_prompt('person',state=state)
print('STEP 8 set_text_prompt done', flush=True)
print('scores_len', len(state.get('scores') or []), flush=True)
