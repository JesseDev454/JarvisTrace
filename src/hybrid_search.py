"""Deterministic window-level reciprocal rank fusion of separate retrieval spaces."""
from __future__ import annotations

import math


def filter_windows_by_similarity(rows, *, mode='hybrid', min_visual_similarity=None,
                                 min_text_similarity=None):
    """Keep windows that clear at least one configured, enabled channel floor.

    Floors are raw channel similarities, never the reciprocal-rank-fusion score.
    With no floors, retain the legacy ranked-candidate behavior.
    """
    if mode not in ('visual', 'text', 'hybrid'):
        raise ValueError('Unknown search mode')
    thresholds = (min_visual_similarity, min_text_similarity)
    for value in thresholds:
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not -1 <= value <= 1
        ):
            raise ValueError('Similarity thresholds must be finite numbers from -1 to 1')
    enabled = []
    if mode in ('visual', 'hybrid') and min_visual_similarity is not None:
        enabled.append(('visual', min_visual_similarity))
    if mode in ('text', 'hybrid') and min_text_similarity is not None:
        enabled.append(('text', min_text_similarity))
    if min_visual_similarity is None and min_text_similarity is None:
        return list(rows)
    if not enabled:
        return []
    accepted = []
    for row in rows:
        for channel, floor in enabled:
            score = row.get('channels', {}).get(channel, {}).get('best_similarity')
            if isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score) and score >= floor:
                accepted.append(row)
                break
    return accepted


def fuse_windows(visual_rows,text_rows,*,window_seconds=4.0,rrf_k=60,
                 visual_weight=1.0,text_weight=1.0,start_seconds=None,end_seconds=None,limit=10,
                 min_visual_similarity=None,min_text_similarity=None):
    numbers=(window_seconds,rrf_k,visual_weight,text_weight)
    if (not all(math.isfinite(x) for x in numbers) or window_seconds<=0 or rrf_k<0
            or min(visual_weight,text_weight)<0 or visual_weight+text_weight<=0
            or type(limit) is not int or not 1<=limit<=100):
        raise ValueError('Invalid window/fusion settings')
    filter_windows_by_similarity([],mode='hybrid',min_visual_similarity=min_visual_similarity,
                                 min_text_similarity=min_text_similarity)
    for v in (start_seconds,end_seconds):
        if v is not None and (not math.isfinite(v) or v<0):
            raise ValueError('Time bounds must be finite and nonnegative')
    if start_seconds is not None and end_seconds is not None and end_seconds<=start_seconds:
        raise ValueError('End must follow start')
    windows,profiles={},{}
    qualified_windows=set()
    configured_thresholds=(min_visual_similarity is not None or min_text_similarity is not None)
    for channel,rows,weight in [('visual',visual_rows,visual_weight),('text',text_rows,text_weight)]:
        if weight==0:
            continue
        threshold=min_visual_similarity if channel=='visual' else min_text_similarity
        found={}
        for original_rank,row in enumerate(rows,1):
            duration=row['duration_seconds']
            if not math.isfinite(duration) or duration<=0 or not math.isfinite(row['similarity']):
                raise ValueError('Invalid source duration/similarity')
            if threshold is not None and row['similarity']<threshold:
                continue
            source=(row['video_id'],row['camera_id'],row['source_sha256'])
            profile=(row['source_video_path'],duration)
            if source in profiles and profiles[source]!=profile:
                raise ValueError('Conflicting source metadata')
            profiles[source]=profile
            if channel=='visual':
                t=row['timestamp_seconds']
                if not math.isfinite(t) or not 0<=t<duration:
                    raise ValueError('Invalid observation time')
                if ((start_seconds is not None and t<start_seconds) or
                        (end_seconds is not None and t>=end_seconds)):
                    continue
                bins=[math.floor(t/window_seconds)]
            else:
                a,b=row['start_seconds'],row['end_seconds']
                if not all(math.isfinite(x) for x in (a,b)) or not 0<=a<b<=duration:
                    raise ValueError('Invalid description time range')
                a=max(a,start_seconds or 0)
                b=min(b,end_seconds if end_seconds is not None else duration)
                if a>=b:
                    continue
                first,last=math.floor(a/window_seconds),math.floor(math.nextafter(b,-math.inf)/window_seconds)
                if last-first>10000:
                    raise ValueError('Description spans too many windows; increase window size')
                bins=range(first,last+1)
            for bucket in bins:
                key=(*source,bucket)
                a=max(bucket*window_seconds,start_seconds or 0)
                b=min((bucket+1)*window_seconds,duration,end_seconds if end_seconds is not None else duration)
                if a>=b:
                    continue
                if key not in found:
                    found[key]={'original_rank':original_rank,'evidence':[]}
                if len(found[key]['evidence'])<5:
                    found[key]['evidence'].append(dict(row))
                if threshold is not None:
                    qualified_windows.add(key)
                if key not in windows:
                    windows[key]=dict(video_id=row['video_id'],camera_id=row['camera_id'],
                        source_sha256=row['source_sha256'],source_video_path=row['source_video_path'],
                        start_seconds=a,end_seconds=b,rrf_score=0.0,channels={})
        # Collapse repeated frame/crop/description hits before assigning channel ranks.
        for rank,(key,hit) in enumerate(sorted(found.items(),key=lambda pair:(pair[1]['original_rank'],pair[0])),1):
            contribution=weight/(rrf_k+rank)
            windows[key]['rrf_score']+=contribution
            windows[key]['channels'][channel]=dict(rank=rank,weight=weight,
                rrf_contribution=contribution,best_similarity=hit['evidence'][0]['similarity'],
                evidence=hit['evidence'])
    if configured_thresholds:
        windows={key:row for key,row in windows.items() if key in qualified_windows}
    return sorted(windows.values(),key=lambda w:(-w['rrf_score'],w['video_id'],w['camera_id'],
                  w['source_sha256'],w['start_seconds']))[:limit]
