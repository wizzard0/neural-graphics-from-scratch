# Instant Neural Graphics From Scratch

The idea:
1. Implement Karpathy's [micrograd] in JS (about 10,000,000x slower than e.g. pytorch on GPU)
2. Implement multiresolution hash encoding from nVIDIA's [Instant Neural Graphics Primitives] 
which is claimed to be 1,000x faster than e.g. [NeRF] or other neural rendering approaches
3. Start with a 64x64 not 20000x24000 pixel image
4. We got a self-contained practice environment to learn neural rendering, well under 500 lines of code!
5. Oh it works o_O let's try some bigger images

Also this is an actual neural network image compression, btw (esp. if you quantize the weights).  
What a time to be alive.

## Results

| Plain MLP                      | 1-level NGP                            | Real NGP                               | More params                                        | NGP + Fancier MLP                   |
|--------------------------------|----------------------------------------|----------------------------------------|----------------------------------------------------|-------------------------------------|
| 64x64px, 10 **hours**              | 8x8, 10 **seconds**                    | ~10min, 1.3k params for 16k pix        | ~15min, 3.3k params, 16k px                        | 40 neurons not 20, 4k params, 15min |
| ![p3817.gif](nice%2Fp3817.gif) | ![naive-ngp.gif](nice%2Fnaive-ngp.gif) | ![16k-1341p.gif](nice%2F16k-1341p.gif) | ![16k-3.3k-params.gif](nice%2F16k-3.3k-params.gif) |  ![16k-4.4k-mlp.gif](nice%2F16k-4.4k-mlp.gif)                               |


## Open questions

- What loss function would preserve those background details?
- Why ReLU is so unstable on this task, even with very low training rates? (see below)

 
![unstable-relu.gif](nice%2Funstable-relu.gif)

## image generation utils

```bash
bun instant-ngp.ts
```
sorry everything's hardcoded rn -_-

gif 64x64
```bash
ffmpeg -pattern_type glob -i 'out/all/loopC-*.png' -vf "crop=64:128:0:0" -r 10 out/output.gif -y
```
gif 128x128
```bash
ffmpeg -pattern_type glob -i 'out/all/loopC-*.png' -vf "crop=128:256:0:0" -r 10 out/output.gif -y
```
cleanup
```bash
rm out/all/loopC-*.png out/10/loopC-*.png out/100/loopC-*.png
```

## credits:

- [micrograd] by Andrej Karpathy
- [Instant Neural Graphics Primitives] by nVIDIA
- cameraman from the USC-SIPI image database
- all errors mine of course (**@oleksandr_now** on [twitter]/[telegram])

[micrograd]: https://github.com/karpathy/micrograd
[Instant Neural Graphics Primitives]: https://github.com/NVlabs/instant-ngp
[NeRF]: https://www.matthewtancik.com/nerf
[twitter]: https://twitter.com/oleksandr_now
[telegram]: https://t.me/oleksandr_now
