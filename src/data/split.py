import pandas as pd
import logging
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)

def group_aware_split(manifest_df, config):
    set_seed(config.get('seed', 42))
    train_size = config.get('train_split', 0.8)
    
    gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=config.get('seed', 42))
    
    X = manifest_df
    y = manifest_df['label']
    groups = manifest_df['source_video']
    
    train_idx, temp_idx = next(gss.split(X, y, groups))
    
    train_df = manifest_df.iloc[train_idx].copy()
    temp_df = manifest_df.iloc[temp_idx].copy()
    
    # Split temp into val and test
    val_size_relative = 0.5 # assuming val=0.1, test=0.1 from remaining 0.2
    gss_temp = GroupShuffleSplit(n_splits=1, train_size=val_size_relative, random_state=config.get('seed', 42))
    
    val_idx, test_idx = next(gss_temp.split(temp_df, temp_df['label'], temp_df['source_video']))
    
    val_df = temp_df.iloc[val_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()
    
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    
    result_df = pd.concat([train_df, val_df, test_df])
    
    _print_split_stats(result_df)
    
    return result_df

def simple_split(manifest_df, config):
    set_seed(config.get('seed', 42))
    train_size = config.get('train_split', 0.8)
    
    train_df, temp_df = train_test_split(
        manifest_df, train_size=train_size, stratify=manifest_df['label'], random_state=config.get('seed', 42)
    )
    
    val_df, test_df = train_test_split(
        temp_df, train_size=0.5, stratify=temp_df['label'], random_state=config.get('seed', 42)
    )
    
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    
    result_df = pd.concat([train_df, val_df, test_df])
    
    _print_split_stats(result_df)
    
    return result_df

def split_dataset(manifest_df, config):
    if 'source_video' in manifest_df.columns and not manifest_df['source_video'].isnull().all():
        logger.info("Using group_aware_split because source_video is present.")
        return group_aware_split(manifest_df, config)
    else:
        logger.info("Using simple_split because source_video is not present or entirely null.")
        return simple_split(manifest_df, config)

def _print_split_stats(df):
    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split]
        logger.info(f"Split: {split}")
        logger.info(f"  Total images: {len(split_df)}")
        if 'source_video' in split_df.columns and not split_df['source_video'].isnull().all():
            logger.info(f"  Source videos: {split_df['source_video'].nunique()}")
        
        real_count = len(split_df[split_df['label'] == 0]) # REAL = 0
        fake_count = len(split_df[split_df['label'] == 1]) # DEEPFAKE = 1
        logger.info(f"  REAL: {real_count}, DEEPFAKE: {fake_count}")
