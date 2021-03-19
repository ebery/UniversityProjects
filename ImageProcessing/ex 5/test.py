import matplotlib.pyplot as plt
import sol5

'''
x = sol5.np.arange(1, 6)

denoising_val_loss = []
deblurring_val_loss = []

for i in range(1, 6):
    model = sol5.learn_denoising_model(num_res_blocks=i, quick_mode=True)
    denoising_val_loss.append(model.history.history['val_loss'][-1])

for i in range(1, 6):
    model = sol5.learn_deblurring_model(num_res_blocks=i, quick_mode=True)
    deblurring_val_loss.append(model.history.history['val_loss'][-1])


print(denoising_val_loss)
print(deblurring_val_loss)

plt.figure()
plt.title('Depth Plot Denoise')
plt.xlabel('Residual Blocks Number')
plt.ylabel('Validation Loss')
plt.plot(x, denoising_val_loss, c='blue', marker='.', mfc='r', ms=10, lw=.5)
# plt.savefig('depth_plot_denoise.png')

plt.figure()
plt.title('Depth Plot Deblur')
plt.xlabel('Residual Blocks Number')
plt.ylabel('Validation Loss')
plt.plot(x, deblurring_val_loss, c='blue', marker='.', mfc='r', ms=10, lw=.5)
# plt.savefig('depth_plot_deblur.png')

plt.show()
'''

noise_im = sol5.read_image('examples/birds_corrupted.png', 1)
noise_orig = sol5.read_image('examples/birds_original.png', 1)
noise_model = sol5.learn_denoising_model()
restored_noise = sol5.restore_image(noise_im, noise_model)
fig1, ax1 = plt.subplots(1, 2)
ax1[0].imshow(noise_orig, cmap='gray')
ax1[0].title.set_text('Original')
ax1[1].imshow(restored_noise, cmap='gray')
ax1[1].title.set_text('fixed')


blur_im = sol5.read_image('examples/text_corrupted.png', 1)
blur_orig = sol5.read_image('examples/text_original.png', 1)
blur_model = sol5.learn_deblurring_model()
restored_blur = sol5.restore_image(blur_im, blur_model)
fig2, ax2 = plt.subplots(1, 2)
ax2[0].imshow(blur_orig, cmap='gray')
ax2[0].title.set_text('Original')
ax2[1].imshow(restored_blur, cmap='gray')
ax2[1].title.set_text('fixed')


plt.show()
