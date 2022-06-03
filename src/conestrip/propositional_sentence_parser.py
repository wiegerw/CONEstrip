#!/usr/bin/env python3

from fractions import Fraction
import z3

import tpg


class PropositionalSentenceParser(tpg.Parser):
    r"""
           separator space '\s+' ;

           token ID     '[a-zA-Z]' ;
           token ZERO   '0' ;
           token ONE    '1' ;
           token LPAREN '\(' ;
           token RPAREN '\)' ;
           token EQUIV  '==' ;
           token COMMA ',' ;

           START/x -> Formula/x ;

           FormulaList/l
                  ->
                     Formula/x                        $ l = [x] $
                     (
                       COMMA Formula/x                $ l.append(x) $
                     )*
                    ;

           Formula/x -> ZERO                                                $ x = False
                      | ONE                                                 $ x = True
                      | 'And' LPAREN FormulaList/l RPAREN                   $ x = z3.And(l)
                      | 'Or' LPAREN FormulaList/l RPAREN                    $ x = z3.Or(l)
                      | 'Not' LPAREN Formula/x1 RPAREN                      $ x = z3.Not(x1)
                      | 'Eq' LPAREN Formula/x1 COMMA Formula/x2 RPAREN      $ x = (x1 == x2)
                      | 'Implies' LPAREN Formula/x1 COMMA Formula/x2 RPAREN $ x = z3.Implies(x1, x2)
                      | ID/x1                                               $ x = z3.Bool(x1)
                      ;
    """


if __name__ == "__main__":
    print("PropositionalSentenceParser (TPG example)")
    parser = PropositionalSentenceParser()
    while 1:
        text = input("\n:")
        if text:
            try:
                print(parser(text))
            except Exception:
                print(tpg.exc())
        else:
            break
