
import os
import argparse
import torch
import torch.nn as nn

import model
import utils

def get_args():
	parse = argparse.ArgumentParser()
	parse.add_argument('--datasets', type=str, default='CIFAR', choices=['STL', 'CIFAR', 'OWN'])
	parse.add_argument('--data_path', type=str, default='./data')
	parse.add_argument('--model_path', type=str, default='./model')
	parse.add_argument('--model_name', type=str, default='model.pth')
	parse.add_argument('--img_size', type=int, default=128)
	parse.add_argument('--batch_size', type=int, default=32)

	parse.add_argument('--epoch', type=int, default=30)
	parse.add_argument('--log_step', type=int, default=10)
	parse.add_argument('--lr',type=float, default=0.0001)
	parse.add_argument('-s', '--save_model_in_epoch', action='store_true')
	config = parse.parse_args()
	print(config)
	return config

def train():
	config = get_args()
	if not os.path.exists(config.model_path):
		os.mkdir(config.model_path)

	train_loader, num_classes = utils.get_train_loader(config.data_path,
										config.datasets, 
										config.img_size,
										config.batch_size)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	cnn = model.CNN(img_size=config.img_size, num_classes=num_classes).to(device)
	print(cnn)
	criterion = nn.CrossEntropyLoss().to(device)

	optimizer = torch.optim.Adam(cnn.parameters(), config.lr)
	min_loss = 999
	print("----------start training----------")

	for epoch in range(config.epoch):
		epoch_loss = 0
		for i, (images, labels) in enumerate(train_loader):
			images, labels = images.to(device), labels.to(device)
			optimizer.zero_grad()
			outputs, _ = cnn(images)
			loss = criterion(outputs, labels)
			optimizer.step()

			epoch_loss += loss.item()

			if (i+1) % config.log_step == 0:
				if config.save_model_in_epoch:
					torch.save(cnn.state_dict(), os.path.join(config.model_path, config.model_name))
				print('Epoch [%d/%d], Iter [%d/%d], Loss: .4F' % (epoch+1, config.epoch, i+1, len(train_loader), loss.item()))
		
		avg_epoch_loss = epoch_loss / len(train_loader)
		print('Epoch [%d/%d], Loss: %.4f' %(epoch+1, config.epoch, avg_epoch_loss))

		if avg_epoch_loss <min_loss:
			min_loss = avg_epoch_loss
			torch.save(cnn.state_dict(), os.path.join(config.model_path, config.model_name))


if __name__ == '__main__':

	train()















