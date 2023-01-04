declare name		"matrix_mixer";
declare version 	"1.0";
declare author 		"AmbisonicDecoderToolkit";
declare license 	"GPL";
declare copyright	"(c) Aaron J. Heller 2020";
declare options "[http:on]";

// bus
bus(n) = par(j, n, _);

// bus with gains
gain(c) = R(c) with {
  R((c,cl)) = R(c),R(cl);
  R(1)      = _;
  R(0)      = !;
  //R(0)      = !:0; // if you need to preserve the number of outputs
  R(float(0)) = R(0);
  R(float(1)) = R(1);
  R(c)      = *(c);
};

// https://faust.grame.fr/doc/manual/#nentry-primitive
// https://faust.grame.fr/doc/manual/#variable-parts-of-a-label
// https://faust.grame.fr/doc/manual/#labels-as-pathnames
// https://faust.grame.fr/doc/manual/#ordering-ui-elements

// n = number of inputs
// m = number of output
matrix(n, m) = bus(n)
               <: par(i, m,
	              hgroup("[%i]out-%i",
	                      gain(par(j, n, nentry("[%j]in-%j[hidden:1]",
			                            0, -1, +1, 0.0001))) :> _
			     )
		      )
	       : bus(m);

process = matrix(49, 64);
//process = matrix(13, 11);
//process = matrix(4, 6);


