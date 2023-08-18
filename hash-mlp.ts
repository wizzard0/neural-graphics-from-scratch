import {FullyConnectedLayer} from "./layer.js";
import {NgpGridLayer} from "./hash-grid.js";
import {Value} from "./value.js";
import {Layer} from "./base.js";

export class NgpMLP {
  layers: Layer[];

  constructor(nin: number, params: number[], sizes: number[], mlpParams: number[]) {
    if (nin != 2) {throw new Error("only 2d input supported");}
    this.layers = [
      new NgpGridLayer(params, sizes, 2),
    ];
    nin = nin + 2 * sizes.length;
    for (let nout of mlpParams) {
      this.layers.push(new FullyConnectedLayer(nin, nout));
      nin = nout;
    }
  }

  forward(x: number[] | Value[]): Value[] | Value {
    for (let layer of this.layers) {
      //console.log({forward:layer.constructor.name,inputs:x.length,layer:[layer.nin,layer.nout]})
      //@ts-ignore
      x = layer.forward(x);
    }
    if (Array.isArray(x) && x.length == 1) {return x[0] as Value;}
    return x as Value[];
  }

  parameters(): Value[] {
    return this.layers.flatMap(l => l.parameters());
  }
}