import png from "./vendor/pnglib-es6.js";
import fs from "fs";
import {MLP} from "./mlp.js";
import {Value} from "./value.js";

export let PNGImage = png;

export interface ThePNGImage {
  width: number;
  height: number;
  depth: number; // bits per pixel
  color(r: number, g: number, b: number, a: number): number; // returns color index
  createColor(color: string): number;
  setPixel(x: number, y: number, color: number): void;
  getPixel(x: number, y: number): number; // returns color index
  buffer: Uint8Array // should access underlying buffer
}

export function saveImage(image: ThePNGImage, file: string) {
  fs.writeFileSync(file, new Uint8Array(image.buffer.buffer));
}

export const dim = 128;

export function clamp(x: number, min: number, max: number) {
  return Math.min(max, Math.max(min, x));
}

export function render(nn: MLP, grays: number[], image: ThePNGImage) {
  for (let y = 0; y < dim; y++) {
    for (let x = 0; x < dim; x += 1) {
      let input = [x / dim, y / dim]; // normalized
      let output = nn.forward(input);
      let brightness = ((output as Value).data)// + ofs) / range;
      // minColor = Math.min(minColor, brightness);
      // maxColor = Math.max(maxColor, brightness);
      //let color = brightness>0.5?white:black;
      let color = grays[clamp(Math.floor(brightness * 256), 0, 255) | 0];
      image.setPixel(x, y, color);
    }
  }
//  console.log({minColor,maxColor});
}