data_transforms = {
    'train': transforms.Compose([
        rightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
        transforms.ToTensor()
    ]),
    'val': transforms.Compose([
        transforms.ToTensor()
    ]),
}

data_dir = '/content/tiny-imagenet-200'

val_label_dir = '/content/tiny-imagenet-200/val/val_annotations.txt'

image_datasets = {}

image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                          data_transforms['train'])

image_datasets['val']=CustomImageDataset(val_label_dir, data_dir+'/val/images',
                                           os.path.join(data_dir, 'train'),
                                           transform = data_transforms['val'])

dataset_sizes={x: len(image_datasets[x]) for x in ['train', 'val']}

dataloaders={x: torch.utils.data.DataLoader(image_datasets[x], batch_size = 4,
                                             shuffle = False, num_workers = 2)
                                            for x in ['train', 'val']}
images = None
for i, (inputs, labels) in enumerate(dataloaders['train']):
    images = inputs
    break

fig = plt.figure(figsize=(15, 10))
rows = 1
columns = 4
image0 = np.transpose(images[0].numpy(), (1, 2, 0))
image1 = np.transpose(images[1].numpy(), (1, 2, 0))
image2 = np.transpose(images[2].numpy(), (1, 2, 0))
image3 = np.transpose(images[3].numpy(), (1, 2, 0))

fig.add_subplot(rows, columns, 1)
plt.imshow(image0)
plt.axis('off')
plt.title('First')

fig.add_subplot(rows, columns, 2)
plt.imshow(image1)
plt.axis('off')
plt.title('Second')

fig.add_subplot(rows, columns, 3)
plt.imshow(image2)
plt.axis('off')
plt.title('Third')

fig.add_subplot(rows, columns, 4)
plt.imshow(image3)
plt.axis('off')
plt.title('Forth')

plt.show()

fig = plt.figure(figsize=(15, 10))
rows = 1
columns = 4
fig.add_subplot(rows, columns, 1)
plt.imshow(img1)
plt.axis('off')
plt.title('eps=0.02')

fig.add_subplot(rows, columns, 2)
plt.imshow(img2)
plt.axis('off')
plt.title('eps=0.05')

fig.add_subplot(rows, columns, 3)
plt.imshow(img3)
plt.axis('off')
plt.title('eps=0.1')

fig.add_subplot(rows, columns, 4)
plt.imshow(img4)
plt.axis('off')
plt.title('eps=0.2')

plt.show()
