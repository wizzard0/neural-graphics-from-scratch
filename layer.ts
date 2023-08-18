import {Value} from "./value.js";
import {Neuron} from "./neuron.js";
import {Layer} from "./base.js";

export class FullyConnectedLayer implements Layer {
  neurons: Neuron[];
  flat: Value[];

  constructor(nin: number, nout: number) {
    this.neurons = [];
    for (let i = 0; i < nout; i++) {
      this.neurons.push(new Neuron(nin));
    }
    this.flat = this.neurons.flatMap(n => n.parameters());
  }

  forward(x: number[] | Value[]): Value[] {
    return this.neurons.map(n => n.forward(x));
  }

  parameters(): Value[] {
    return this.flat;
  }
}