from datasets.split import splits_2020 as splits
from datasets.osr_dataloader import MNIST_OSR, CIFAR10_OSR, CIFAR100_OSR, SVHN_OSR, Tiny_ImageNet_OSR

def get_dataloader(options):
    fold = options['fold']
    known = splits[options['dataset']][len(splits[options['dataset']])- fold -1]
    if options['dataset'] == 'cifar100':
        unknown = splits[options['dataset']+'-'+str(options['out_num'])][len(splits[options['dataset']])-fold-1]
    elif options['dataset'] == 'tiny_imagenet':
        #img_size = 64
        #options['lr'] = 0.001
        unknown = list(set(list(range(0, 200))) - set(known))
    else:
        unknown = list(set(list(range(0, 10))) - set(known))

    options.update(
        {
            'known':    known,
            'unknown':  unknown,
            #'img_size': img_size
        }
    )

    print("{} Preparation".format(options['dataset']))
    if 'mnist' in options['dataset']:
        Data = MNIST_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    elif 'cifar10' == options['dataset']:
        Data = CIFAR10_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    elif 'svhn' in options['dataset']:
        Data = SVHN_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    elif 'cifar100' in options['dataset']:
        Data = CIFAR10_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
        trainloader, testloader = Data.train_loader, Data.test_loader
        out_Data = CIFAR100_OSR(known=options['unknown'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
        outloader = out_Data.test_loader
    else:
        Data = Tiny_ImageNet_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    
    options['num_classes'] = Data.num_classes

    dataloader = {'train': trainloader, 'test': testloader, 'ood': outloader}
    return dataloader, options