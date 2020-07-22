clc
clear
close all
%%

%%

device01=load('N1M2.mat','-ascii');
device02=load('N2M2.mat','-ascii');
device03=load('N3M2.mat','-ascii');

%load mnist_uint8;
train_x = device02(1:3000,:);

train_x = mapminmax(train_x, 0, 1);

%train_y_normal = double(ones(size(train_x,1),1));
%train_y_spoofing= double(zeros(size(test_x_spoofing,1),1));
%train_y_all=[train_y_normal;train_y_spoofing];

% real label [1 0]  general label [0 1];
train_y = double(ones(size(train_x,1),1));
% normalize
%train_x = mapminmax(train_x, 0, 1);
rand('state',0)
%% 
noise_d=100;
batch_size = 100;
epoch = 200;
images_num = 3000;
batch_num = ceil(images_num / batch_size);
learning_rate = 0.012;

%for t_n=1:50
generator = nnsetup1([noise_d, 100, 36]);
discriminator = nnsetup1([36, 100, 1]);

for e=1:epoch
    kk = randperm(images_num);
    for t=1:batch_num
        % 准备数据
        images_real = train_x(kk((t - 1) * batch_size + 1:t * batch_size), :, :);
        noise = unifrnd(-1, 1, batch_size, noise_d);
        % 开始训练
        % -----------更新generator，固定discriminator
        generator = nnff1(generator, noise);
        images_fake = generator.layers{generator.layers_count}.a;
        discriminator = nnff1(discriminator, images_fake);
        logits_fake = discriminator.layers{discriminator.layers_count}.z;
        discriminator = nnbp_d1(discriminator, logits_fake, ones(batch_size, 1));
        generator = nnbp_g1(generator, discriminator);
        generator = nnapplygrade1(generator, learning_rate);
        % -----------更新discriminator，固定generator
        generator = nnff1(generator, noise);
        images_fake = generator.layers{generator.layers_count}.a;
        images = [images_fake;images_real];
        discriminator = nnff1(discriminator, images);
        logits = discriminator.layers{discriminator.layers_count}.z;
        labels = [zeros(batch_size,1);ones(batch_size,1)];
        discriminator = nnbp_d1(discriminator, logits, labels);
        discriminator = nnapplygrade1(discriminator, learning_rate);
        % ----------------输出loss
        if t == batch_num
            c_loss = sigmoid_cross_entropy1(logits(1:batch_size), ones(batch_size, 1));
            d_loss = sigmoid_cross_entropy1(logits, labels);
            fprintf('c_loss:"%f",d_loss:"%f"\n',c_loss, d_loss);
        end

    end

end

noise_data = 3000;
noise = unifrnd(-1, 1, noise_data, noise_d);
generator = nnff1(generator, noise);
gan_data = generator.layers{generator.layers_count}.a;


save('gan_data.mat','gan_data');
dim=1:36;

figure 
plot(dim,train_x(1:1000,:));
figure 
plot(dim,images_fake);

figure 
plot(dim,gan_data(1:1000,:));

