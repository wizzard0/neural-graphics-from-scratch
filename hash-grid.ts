import {ensureValue, Param, Value} from "./value.js";
import {Layer, random} from "./base.js";

const ySkip = 153499;
const xSkip = 153511;

export class NgpGridLayer implements Layer {
  // picks a single neuron from a Nin-dimensional grid of neurons and extracts nout values from it. no interpolation
  cells0: Param[][] // LOD x T (nParams), N-out is hardcoded to 2
  cells1: Param[][] // LOD x T (nParams)
  nout: number;
  sizes: number[]; // square only for now
  nParams: number[]; // T
  flat: Param[]; // all params in one array

  constructor(nParams: number[],sizes:number[], nout: number) {
    if(nParams.length!=sizes.length){throw new Error("nParams and sizes must have same length");}
    if(nout!=2){throw new Error("n-out must be 2");}
    if(nParams[0]<sizes[0]*sizes[0]){throw new Error("nParams[0] must be at least sizes[0]*sizes[0]");}
    this.nParams = nParams;
    this.nout = nout;
    this.sizes = sizes;
    this.cells0 = [];
    this.cells1 = [];
    // if (inDims.length != 2 || nout != 2 + 10) {
    //   console.log({inDims, nout})
    //   throw new Error("only 2 dims and 2+10 out dims are supported");
    // }
    let lod0:Param[],lod1:Param[]; // level 0 is size*size (coarsest)
    let isc = 0.1;
    for (let level = 0; level < nParams.length; level++) {
      this.cells0.push(lod0=[]);this.cells1.push(lod1=[]);
      for (let y = 0; y < nParams[level]; y++) {
        lod0.push(new Param(random() * isc, ''));
        lod1.push(new Param(random() * isc, ''));
      }
    }
    this.flat = this.cells0.flat().concat(this.cells1.flat());
  }

  index0(xf: number, yf: number) { // without hashing
    if (typeof xf !== 'number' || typeof yf !== 'number') {throw new Error("xf and yf must be numbers");}
    if (xf < 0 || xf > 1 || yf < 0 || yf > 1) { throw new Error("out of bounds, should be normalized");}
    let [width, height] = [this.sizes[0],this.sizes[0]];
    let crop = this.nParams[0];
    let x = xf * width | 0;
    let y = yf * height | 0;
    let lx = xf * width - x;
    let ly = yf * height - y;
    let i00 = (y * width + x) % crop;
    let i01 = (y * width + x + 1) % crop;
    let i10 = ((y + 1) * width + x) % crop;
    let i11 = ((y + 1) * width + x + 1) % crop;
    return [i00, i01, i10, i11, lx, ly];
  }

  hashedIndex(xf: number, yf: number, level: number) {
    if (typeof xf !== 'number' || typeof yf !== 'number') {throw new Error("xf and yf must be numbers");}
    if (xf < 0 || xf > 1 || yf < 0 || yf > 1) { throw new Error("out of bounds, should be normalized");}
    let [width, height] = [this.sizes[level],this.sizes[level]];
    let crop = this.nParams[level];
    let x = xf * width | 0;
    let y = yf * height | 0;
    let lx = xf * width - x;
    let ly = yf * height - y;
    let i00 = ((y * ySkip) ^ (x * xSkip)) % crop;
    let i10 = (((y + 1) * ySkip) ^ (x * xSkip)) % crop;
    let i01 = ((y * ySkip) ^ ((x + 1) * xSkip)) % crop;
    let i11 = (((y + 1) * ySkip) ^ ((x + 1) * xSkip)) % crop;
    return [i00, i01, i10, i11, lx, ly];
  }

  interpolateNearest(cells:Param[], index:number): Value {
    return cells[index % cells.length];
  }

  interpolateLinear(cells: Value[], indices: number[]): Value {
    let [i00, i01, i10, i11, lx, ly] = indices;
    let c00 = cells[i00];
    let c01 = cells[i01];
    let c10 = cells[i10];
    let c11 = cells[i11];
    if (!c00 || !c01 || !c10 || !c11) {
      console.log({indices})
      throw new Error("out of bounds");
    }
    let lerp00 = (1 - ly) * (1 - lx);
    let lerp01 = (1 - ly) * (lx);
    let lerp10 = (ly) * (1 - lx);
    let lerp11 = (ly) * (lx);
    let v00 = c00.mul(lerp00);
    let v01 = c01.mul(lerp01);
    let v10 = c10.mul(lerp10);
    let v11 = c11.mul(lerp11);
    return v00.add(v01).add(v10).add(v11);
  }

  forward(x: number[] | Value[]): Value[] {
    // assume x is normalized, passthrough x as well
    let [x1, y1] = x as number[];
    let outputs: Value[] = new Array(this.sizes.length*2+2); // 2 outputs per level + 2 inputs pass-through
    let indices0 = this.index0(x1, y1);
    outputs[0] = this.interpolateLinear(this.cells0[0], indices0);
    outputs[1] = this.interpolateLinear(this.cells1[0], indices0);

    for(let level=1;level<this.sizes.length-1;level++){
      let indices = this.hashedIndex(x1, y1, level);
      outputs[level*2] = this.interpolateLinear(this.cells0[level], indices);
      outputs[level*2+1] = this.interpolateLinear(this.cells1[level], indices);
    }

    if(this.sizes.length>1) {
      let indicesLast = this.hashedIndex(x1, y1, this.sizes.length - 1);
      outputs[outputs.length - 4] = this.interpolateNearest(this.cells0[this.cells0.length - 1], indicesLast[0]);
      outputs[outputs.length - 3] = this.interpolateNearest(this.cells1[this.cells1.length - 1], indicesLast[0]);
    }

    outputs[outputs.length-2] = ensureValue(x1);
    outputs[outputs.length-1] = ensureValue(y1);
    return outputs;
  }

  parameters(): Value[] {
    return this.flat;
  }
}