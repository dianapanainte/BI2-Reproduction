import os
import os.path as osp
import shutil
import pickle
import torch
from tqdm import tqdm
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_zip)

class EventDatasetBase(InMemoryDataset):
    def __init__(self, root, dataset_name, url, folder_name,
                 split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        """
        Generic dataset class for all event graph datasets
        """
        self.name = dataset_name
        self.url = url
        self.folder_name = folder_name
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)     

    @property
    def raw_file_names(self):
        return ['train.pickle', 'val.pickle', 'test.pickle']
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']
    
    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, self.folder_name), self.raw_dir)
        os.unlink(path)   

    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)
            indices = range(len(graphs))
            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')
            data_list = []
            for idx in indices:
                graph = graphs[idx]             
                x = graph.x
                edge_attr = graph.edge_attr
                edge_index = graph.edge_index
                y = graph.y
                cid = graph.cid
                pl = graph.pl   
                cluster_id = graph.cluster_id
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                            y=y, cid=cid, pl=pl, cluster_id=cluster_id)    
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)
                pbar.update(1)
            pbar.close()
            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))

class EVENTBPIC15M1(EventDatasetBase):
    def __init__(self, root, split='train', transform=None, pre_transform=None, pre_filter=None):
        super().__init__(
            root=root,
            dataset_name="EVENTBPIC15M1",
            url="https://github.com/keyvan-amiri/PGTNet/raw/main/transformation/bpic15m1_graph_raw.zip",
            folder_name="bpic15m1_graph_raw",
            split=split,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter
        )

class EVENTBPIC15M2(EventDatasetBase):
    def __init__(self, root, split='train', transform=None, pre_transform=None, pre_filter=None):
        super().__init__(
            root=root,
            dataset_name="EVENTBPIC15M2",
            url="https://github.com/keyvan-amiri/PGTNet/raw/main/transformation/bpic15m2_graph_raw.zip",
            folder_name="bpic15m2_graph_raw",
            split=split,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter
        )
        
class EVENTBPIC15M3(EventDatasetBase):
    def __init__(self, root, split='train', transform=None, pre_transform=None, pre_filter=None):
        super().__init__(
            root=root,
            dataset_name="EVENTBPIC15M3",
            url="https://github.com/keyvan-amiri/PGTNet/raw/main/transformation/bpic15m3_graph_raw.zip",
            folder_name="bpic15m3_graph_raw",
            split=split,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter
        )

class EVENTBPIC15M4(EventDatasetBase):
    def __init__(self, root, split='train', transform=None, pre_transform=None, pre_filter=None):
        super().__init__(
            root=root,
            dataset_name="EVENTBPIC15M4",
            url="https://github.com/keyvan-amiri/PGTNet/raw/main/transformation/bpic15m4_graph_raw.zip",
            folder_name="bpic15m4_graph_raw",
            split=split,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter
        )
             
class EVENTBPIC15M5(EventDatasetBase):
    def __init__(self, root, split='train', transform=None, pre_transform=None, pre_filter=None):
        super().__init__(
            root=root,
            dataset_name="EVENTBPIC15M5",
            url="https://github.com/keyvan-amiri/PGTNet/raw/main/transformation/bpic15m5_graph_raw.zip",
            folder_name="bpic15m5_graph_raw",
            split=split,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter
        ) 
            
class EVENTBPIC12(EventDatasetBase):
    def __init__(self, root, split='train', transform=None, pre_transform=None, pre_filter=None):
        super().__init__(
            root=root,
            dataset_name="EVENTBPIC12",
            url="https://github.com/keyvan-amiri/PGTNet/raw/main/transformation/bpi12_graph_raw.zip",
            folder_name="bpi12_graph_raw",
            split=split,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter
        ) 

class EVENTHelpDesk(EventDatasetBase):
    def __init__(self, root, split='train', transform=None, pre_transform=None, pre_filter=None):
        super().__init__(
            root=root,
            dataset_name="EVENTHelpDesk",
            url="https://github.com/keyvan-amiri/PGTNet/raw/main/transformation/helpdesk_graph_raw.zip",
            folder_name="helpdesk_graph_raw",
            split=split,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter
        )   

class EVENTHospital(EventDatasetBase):
    def __init__(self, root, split='train', transform=None, pre_transform=None, pre_filter=None):
        super().__init__(
            root=root,
            dataset_name="EVENTHospital",
            url="https://github.com/keyvan-amiri/PGTNet/raw/main/transformation/hospital_graph_raw.zip",
            folder_name="hospital_graph_raw",
            split=split,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter
        )               

class EVENTSepsis(EventDatasetBase):
    def __init__(self, root, split='train', transform=None, pre_transform=None, pre_filter=None):
        super().__init__(
            root=root,
            dataset_name="EVENTSepsis",
            url="https://github.com/keyvan-amiri/PGTNet/raw/main/transformation/sepsis_graph_raw.zip",
            folder_name="sepsis_graph_raw",
            split=split,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter
        )  