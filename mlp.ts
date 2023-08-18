import {Value} from "./value.js";
import {FullyConnectedLayer} from "./layer.js";
import {Layer} from "./base.js";

export class MLP {
  layers: Layer[];
  constructor(nin: number, nouts: number[]) {
    this.layers = [];
    for (let nout of nouts) {
      this.layers.push(new FullyConnectedLayer(nin,nout));
      nin = nout;
    }
  }

  forward(x: number[]|Value[]): Value[]|Value {
    for (let layer of this.layers) {
      //@ts-ignore
      x = layer.forward(x);
    }
    if(Array.isArray(x) && x.length==1){
      return x[0] as Value;
    }
    return x as Value[];
  }

  parameters():Value[] {
    return this.layers.flatMap(l=>l.parameters());
  }
}
