import sys    

class ProgressBar:
    def __init__( self, T, length, title, tname, 
                  tformat = 'g', out = sys.stdout, silent = False ):
        self.T = T
        self.length = length
        self.title = title
        self.tname = tname
        self.tformat = tformat
        self.out = out
        self.silent = silent
        self.last_length = 0

    def render( self, t ):
        if self.silent: 
            return
        prog = t / self.T   if self.T > 0 else 0
        output = ( '\r{title}: [ {bar:{length}} ] {perc}%, '
                   '{tname} = {t:{tformat}}'.format(
                        title = self.title,
                        bar = '#'*int(self.length*prog), 
                        perc = int(100*prog), 
                        tname = self.tname, 
                        tformat = self.tformat,
                        t = t,
                        length = self.length ) )
        self.out.write( '\r' + ' ' * self.last_length )
        self.out.write( output )
        self.out.flush()
        self.last_length = len(output)
        
    def done( self ):
        if self.silent:
            return
        self.render(self.T)
        self.out.write('\n')
        
#from IPython.core import display
#    
#import math
#
#class ProgressBar:
#    def __init__(self, T):
#        self.T = T
#
#    def render(self, t, clear = True):
#        perc = math.floor( 100 * float(t) / self.T )
#        if clear:
#            display.clear_output()
#        display.display( display.HTML( 
#                    '<progress value="%d" max="%d" style="width:500px"></progress> <strong>%g%%</strong> (%d)' 
#                    % (t,self.T,perc,t) ) )