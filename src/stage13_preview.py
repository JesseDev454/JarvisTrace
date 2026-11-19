"""Reuse Stage 12 UI against an isolated evaluation index, without changing public."""
import argparse
from functools import partial
import os
from pathlib import Path

from src.pg_config import open_database
from src.stage13_index import require_local_tables,validate_schema
from src.stage11 import search
from src.stage12 import PlaybackCatalog,create_app
from src.visual_index import VisualIndex
from src.reviewed_text import ReviewedTextIndex


def create_preview(schema,roots):
    validate_schema(schema)
    require_local_tables(schema)
    with open_database(schema=schema,readonly=True) as connection:
        rows=connection.execute('''SELECT DISTINCT video_id,source_camera_id AS camera_id,
            source_sha256,source_video_path,source_duration_seconds AS duration_seconds
            FROM visual_embeddings WHERE source_camera_id IS NOT NULL''').fetchall()
    catalog=PlaybackCatalog(rows,roots)
    return create_app(catalog=catalog,search_fn=partial(search,
        visual_index=VisualIndex(schema=schema),text_index=ReviewedTextIndex(schema=schema)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--schema',required=True)
    parser.add_argument('--storage-root',type=Path,action='append',required=True)
    parser.add_argument('--port',type=int,default=8771)
    args=parser.parse_args()
    if not 1024<=args.port<=65535:parser.error('Port must be 1024..65535')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    from huggingface_hub import constants
    constants.HF_HUB_OFFLINE=True
    import uvicorn
    uvicorn.run(create_preview(args.schema,args.storage_root),host='127.0.0.1',port=args.port,proxy_headers=False)


if __name__=='__main__':main()
