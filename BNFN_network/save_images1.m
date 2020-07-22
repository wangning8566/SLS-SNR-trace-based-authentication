% 保存图片，便于观察generator生成的images_fake
function save_images1(images, count, path)
    n = size(images, 1);
    row = count(1);
    col = count(2);
    I = zeros(row*28, col*28);
    for i = 1:row
        for j = 1:col
            r_s = (i-1)*28+1;
            c_s = (j-1)*28+1;
            index = (i-1)*col + j;
            pic = reshape(images(index, :), 28, 28);
            I(r_s:r_s+27, c_s:c_s+27) = pic;
        end
    end
    imwrite(I, path);
end