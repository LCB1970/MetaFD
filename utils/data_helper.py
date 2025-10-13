from utils.cfgs import DOMAIN_SETS, FILE_FORMAT
from utils.data_reader import *
from torch.utils.data import DataLoader


def dg_dataloader(args):
    train_transform = transforms.Compose(
        [
            transforms.Resize(args.crop_size),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=90),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(args.crop_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    domain_idx = 0
    if args.dg_type == 'MDG':
        for domain in DOMAIN_SETS[args.dad]:
            if domain == args.target_domain: continue
            train_path = f'{args.data_dir}/{domain}/sample/train'
            val_path = f'{args.data_dir}/{domain}/sample/val'
            train_data = MyDataset(
                data_root=train_path,
                transform=train_transform,
                img_format=FILE_FORMAT[domain],
                label_format=FILE_FORMAT[domain],
                domain_label=domain_idx,
                add_domain_information=True
            )
            val_data = MyDataset(
                data_root=val_path,
                transform=val_transform,
                img_format=FILE_FORMAT[domain],
                label_format=FILE_FORMAT[domain],
                domain_label=domain_idx,
                add_domain_information=True
            )
            try:
                mix_train_dataset = ConcatDataset([source_train_loader.dataset, train_data])
                mix_val_dataset = ConcatDataset([source_val_loader.dataset, val_data])
            except:
                mix_train_dataset = train_data
                mix_val_dataset = val_data

            source_train_loader = DataLoader(mix_train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
                                             num_workers=0)
            source_val_loader = DataLoader(mix_val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
                                           num_workers=0)
            domain_idx += 1
            # print(domain, len(source_train_loader), len(source_val_loader))

    else:
        train_path = f'{args.data_dir}/{args.source_domain}/sample/train'
        val_path = f'{args.data_dir}/{args.source_domain}/sample/val'
        train_data = MyDataset(
            data_root=train_path,
            transform=train_transform,
            img_format=FILE_FORMAT[args.source_domain],
            label_format=FILE_FORMAT[args.source_domain],
            domain_label=domain_idx,
            add_domain_information=True
        )
        val_data = MyDataset(
            data_root=val_path,
            transform=val_transform,
            img_format=FILE_FORMAT[args.source_domain],
            label_format=FILE_FORMAT[args.source_domain],
            domain_label=domain_idx,
            add_domain_information=True
        )

        source_train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=False,
                                         num_workers=0)
        source_val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True, drop_last=False,
                                       num_workers=0)
        print(args.source_domain, len(source_train_loader), len(source_val_loader))
    return source_train_loader, source_val_loader
